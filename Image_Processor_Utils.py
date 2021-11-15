import os, copy
from _collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tensorflow_addons.image import transform_ops
import tensorflow_probability as tfp

import numpy as np
import SimpleITK as sitk
from skimage import morphology, measure
from scipy.spatial import distance
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
import cv2
from math import floor, ceil

from threading import Thread
from multiprocessing import cpu_count
from queue import *

import time


class ImageProcessor(object):
    def parse(self, *args, **kwargs):
        return args, kwargs


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


def remove_non_liver(annotations, threshold=0.5, max_volume=9999999.0, min_volume=0.0, max_area=99999.0, min_area=0.0,
                     do_3D=True, do_2D=False, spacing=None):
    '''
    :param annotations: An annotation of shape [Z_images, rows, columns]
    :param threshold: Threshold of probability from 0.0 to 1.0
    :param max_volume: Max volume of structure allowed
    :param min_volume: Minimum volume of structure allowed, in ccs
    :param max_area: Max volume of structure allowed
    :param min_area: Minimum volume of structure allowed
    :param do_3D: Do a 3D removal of structures, only take largest connected structure
    :param do_2D: Do a 2D removal of structures, only take largest connected structure
    :param spacing: Spacing of elements, in form of [z_spacing, row_spacing, column_spacing]
    :return: Masked annotation
    '''
    min_volume = min_volume * (10 * 10 * 10)  # cm to mm3
    annotations = copy.deepcopy(annotations)
    annotations = np.squeeze(annotations)
    if not annotations.dtype == 'int':
        annotations[annotations < threshold] = 0
        annotations[annotations > 0] = 1
        annotations = annotations.astype('int')
    if do_3D:
        labels = morphology.label(annotations, connectivity=1)
        if np.max(labels) > 1:
            area = []
            max_val = 0
            for i in range(1, labels.max() + 1):
                new_area = labels[labels == i].shape[0]
                if spacing is not None:
                    volume = np.prod(spacing) * new_area
                    if volume > max_volume:
                        continue
                    elif volume < min_volume:
                        continue
                area.append(new_area)
                if new_area == max(area):
                    max_val = i
            labels[labels != max_val] = 0
            labels[labels > 0] = 1
            annotations = labels
    if do_2D:
        slice_indexes = np.where(np.sum(annotations, axis=(1, 2)) > 0)
        if slice_indexes:
            for slice_index in slice_indexes[0]:
                labels = morphology.label(annotations[slice_index], connectivity=1)
                if np.max(labels) == 1:
                    continue
                area = []
                max_val = 0
                for i in range(1, labels.max() + 1):
                    new_area = labels[labels == i].shape[0]
                    if spacing is not None:
                        temp_area = np.prod(spacing[1:]) * new_area / 100
                        if temp_area > max_area:
                            continue
                        elif temp_area < min_area:
                            continue
                    area.append(new_area)
                    if new_area == max(area):
                        max_val = i
                labels[labels != max_val] = 0
                labels[labels > 0] = 1
                annotations[slice_index] = labels
    return annotations


def extract_main_component(nparray, dist=50, max_comp=2, min_vol=2000):
    '''
    dist in mm
    min_vol in cubic mm
    '''
    # TODO create a dictionnary of the volume per label to filter the keep_id with max_comp

    labels = morphology.label(nparray, connectivity=3)
    temp_img = np.zeros(labels.shape)

    if np.max(labels) > 1:

        volumes = [labels[labels == i].shape[0] for i in range(1, labels.max() + 1)]
        max_val = volumes.index(max(volumes)) + 1

        keep_values = []
        keep_values.append([max_val, max(volumes)])

        if dist:
            ref_volume = np.copy(labels)
            ref_volume[ref_volume != max_val] = 0
            ref_volume[ref_volume > 0] = 1
            temp_volume = ref_volume[ref_volume == 1].shape[0]
            # volume of largest comp is too small
            if temp_volume < min_vol:
                return temp_img
            else:
                try:
                    ref_points = measure.marching_cubes(ref_volume, step_size=3, method='lewiner')[0]
                except:
                    # volume of largest comp is too small and make the marching cube fails
                    print(" Warning: volume of largest comp is too small for marching cube")
                    dist = False

        for i in range(1, labels.max() + 1):
            if i == max_val:
                continue

            # compute distance
            temp_label = np.copy(labels)
            temp_label[temp_label != i] = 0
            temp_label[temp_label > 0] = 1

            temp_volume = temp_label[temp_label == 1].shape[0]

            if temp_volume < min_vol:
                continue

            if dist:
                try:
                    # this remove small 'artifacts' cause they cannot be meshed
                    temp_points = measure.marching_cubes(temp_label, step_size=3, method='lewiner')[0]
                except:
                    continue

                for ref_point in ref_points:
                    distances = [distance.euclidean(ref_point, temp_point) for temp_point in temp_points]
                    if min(distances) < dist:
                        keep_values.append([i, temp_volume])
                        break
            else:
                keep_values.append([i, temp_volume])

        # sort by volume
        keep_values = sorted(keep_values, key=lambda x: x[1], reverse=True)

        # this is faster than 'temp_img[np.isin(labels, keep_values)] = 1' for most cases
        for values in keep_values[0:max_comp]:
            temp_img[labels == values[0]] = 1
        nparray = temp_img

    return nparray


def compute_bounding_box(annotation, padding=2):
    '''
    :param annotation: A binary image of shape [# images, # rows, # cols, channels]
    :return: the min and max z, row, and column numbers bounding the image
    '''
    shape = annotation.shape
    indexes = np.where(np.any(annotation, axis=(1, 2)) == True)[0]
    min_slice, max_slice = max(0, indexes[0] - padding), min(indexes[-1] + padding, shape[0])
    '''
    Get the row values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
    min_row, max_row = max(0, indexes[0] - padding), min(indexes[-1] + padding, shape[1])
    '''
    Get the col values of primary and secondary
    '''
    indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
    min_col, max_col = max(0, indexes[0] - padding), min(indexes[-1] + padding, shape[2])
    return [min_slice, max_slice, min_row, max_row, min_col, max_col]


def compute_binary_morphology(input_img, radius=1, morph_type='closing'):
    # this is faster than using sitk binary morphology filters (dilate, erode, opening, closing)
    if len(input_img.shape) == 2:
        struct = morphology.disk(radius)
    elif len(input_img.shape) == 3:
        struct = morphology.ball(radius)
    else:
        raise ValueError("Dim {} for morphology structure element not supported".format(len(input_img.shape)))

    if morph_type == 'closing':
        input_img = binary_closing(input_img, structure=struct)
    elif morph_type == 'opening':
        input_img = binary_opening(input_img, structure=struct)
    else:
        raise ValueError("Type {} is not supported".format(morph_type))

    return input_img


class compute_binary_metrics(object):
    def __init__(self, input1, input2, img_dtype=sitk.sitkUInt8):
        if isinstance(input1, np.ndarray):
            if input1.shape[-1] == 1:
                input1 = np.squeeze(input1, axis=-1)
            input1 = sitk.GetImageFromArray(input1)

        if isinstance(input2, np.ndarray):
            if input2.shape[-1] == 1:
                input2 = np.squeeze(input2, axis=-1)
            input2 = sitk.GetImageFromArray(input2)

        if input1.GetPixelIDValue() != img_dtype:
            cast_filter = sitk.CastImageFilter()
            cast_filter.SetOutputPixelType(img_dtype)
            input1 = cast_filter.Execute(input1)

        if input2.GetPixelIDValue() != img_dtype:
            cast_filter = sitk.CastImageFilter()
            cast_filter.SetOutputPixelType(img_dtype)
            input2 = cast_filter.Execute(input2)

        self.metric_filter = sitk.LabelOverlapMeasuresImageFilter()
        self.metric_filter.SetNumberOfThreads(0)
        self.metric_filter.Execute(input1, input2)

    def get_dice(self):
        return (self.metric_filter.GetDiceCoefficient())

    def get_jaccard(self):
        return (self.metric_filter.GetJaccardCoefficient())


class compute_distance_metrics(object):
    def __init__(self, input1, input2, img_dtype=sitk.sitkUInt8):
        if isinstance(input1, np.ndarray):
            if input1.shape[-1] == 1:
                input1 = np.squeeze(input1, axis=-1)
            input1 = sitk.GetImageFromArray(input1)

        if isinstance(input2, np.ndarray):
            if input2.shape[-1] == 1:
                input2 = np.squeeze(input2, axis=-1)
            input2 = sitk.GetImageFromArray(input2)

        if input1.GetPixelIDValue() != img_dtype:
            cast_filter = sitk.CastImageFilter()
            cast_filter.SetOutputPixelType(img_dtype)
            input1 = cast_filter.Execute(input1)

        if input2.GetPixelIDValue() != img_dtype:
            cast_filter = sitk.CastImageFilter()
            cast_filter.SetOutputPixelType(img_dtype)
            input2 = cast_filter.Execute(input2)

        self.metric_filter = sitk.HausdorffDistanceImageFilter()
        self.metric_filter.SetNumberOfThreads(0)
        self.metric_filter.Execute(input1, input2)

    def get_hd(self):
        return (self.metric_filter.GetHausdorffDistance())

    def get_avg_hd(self):
        return (self.metric_filter.GetAverageHausdorffDistance())


class Remove_Smallest_Structures(object):
    def __init__(self):
        self.Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        self.Connected_Component_Filter.SetNumberOfThreads(0)
        self.RelabelComponent = sitk.RelabelComponentImageFilter()
        self.RelabelComponent.SortByObjectSizeOn()
        self.RelabelComponent.SetNumberOfThreads(0)

    def remove_smallest_component(self, annotation):
        if type(annotation) is np.ndarray:
            annotation_handle = sitk.GetImageFromArray(annotation.astype(np.uint8))
            convert = True
        else:
            annotation_handle = annotation
        label_image = self.Connected_Component_Filter.Execute(
            sitk.BinaryThreshold(sitk.Cast(annotation_handle, sitk.sitkFloat32), lowerThreshold=0.01,
                                 upperThreshold=np.inf))
        label_image = self.RelabelComponent.Execute(label_image)
        output = sitk.BinaryThreshold(sitk.Cast(label_image, sitk.sitkFloat32), lowerThreshold=0.1, upperThreshold=1.0)
        if convert:
            output = sitk.GetArrayFromImage(output)
        return output


class Fill_Hole_Binary(object):
    def __init__(self):
        self.BinaryFillholeImageFilter = sitk.BinaryFillholeImageFilter()
        self.BinaryFillholeImageFilter.SetNumberOfThreads(0)
        self.BinaryFillholeImageFilter.SetForegroundValue(1)

    def fill_hole_binary_3D(self, annotation):
        if type(annotation) is np.ndarray:
            annotation_handle = sitk.GetImageFromArray(annotation.astype(np.uint8))
            convert = True
        else:
            annotation_handle = annotation

        output = self.BinaryFillholeImageFilter.Execute(annotation_handle)
        if convert:
            output = sitk.GetArrayFromImage(output)
        return output

    def fill_hole_binary_2D(self, annotation):

        if type(annotation) is np.ndarray:
            annotation_handle = sitk.GetImageFromArray(annotation.astype(np.uint8))
            convert = True
        else:
            annotation_handle = annotation

        annotation_shape = annotation.shape
        output_list = []

        # shapes are reversed in ITK
        for index in range(0, annotation_shape[0]):
            extract_filter = sitk.ExtractImageFilter()
            extract_filter.SetSize([annotation_shape[2], annotation_shape[1], 0])
            extract_filter.SetIndex([0, 0, index])
            annotation_slice = extract_filter.Execute(annotation_handle)
            output_slice = self.BinaryFillholeImageFilter.Execute(annotation_slice)
            output_list.append(output_slice)

        join_series_filter = sitk.JoinSeriesImageFilter()
        join_series_filter.SetNumberOfThreads(0)
        output = join_series_filter.Execute(output_list)

        if convert:
            output = sitk.GetArrayFromImage(output)
        return output


class CreateExternal(ImageProcessor):
    def __init__(self, image_key='image', output_key='external', output_type=np.int16, threshold_value=-250.0,
                 mask_value=1, run_3D=False):
        self.image_key = image_key
        self.output_key = output_key
        self.output_type = output_type
        self.threshold_value = threshold_value
        self.mask_value = mask_value
        self.run_3D = run_3D

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=(self.image_key,))
        image = input_features[self.image_key]
        self.external_mask = np.zeros(image.shape, dtype=self.output_type)
        self.external_mask[image > self.threshold_value] = self.mask_value

        # remove small stuff (for example table or mask artefact)
        self.external_mask = morphology.opening(image=self.external_mask, selem=morphology.ball(2))

        # remove unconnected component
        main_component_filter = Remove_Smallest_Structures()
        self.external_mask = main_component_filter.remove_smallest_component(self.external_mask)

        # fill holes per slice (avoid bowel bag missed on top of image)
        fill_hole_filter = Fill_Hole_Binary()
        if self.run_3D:
            self.external_mask = fill_hole_filter.fill_hole_binary_3D(self.external_mask)
        else:
            self.external_mask = fill_hole_filter.fill_hole_binary_2D(self.external_mask)

        input_features[self.output_key] = self.external_mask
        return input_features

    def post_process(self, input_features):
        return input_features


class Focus_on_CT(ImageProcessor):
    def __init__(self, threshold_value=-250.0, mask_value=1, debug=False, annotation=False):
        # TODO this class needs to be cleaned
        self.threshold_value = threshold_value
        self.mask_value = mask_value
        self.bb_parameters = []
        self.final_padding = []
        self.original_shape = {}
        self.squeeze_flag = False
        self.debug = debug
        self.annotation = annotation

    def pre_process(self, input_features):
        images = input_features['image']
        if images.dtype != 'float32':
            images = images.astype('float32')

        if len(images.shape) > 3:
            images = np.squeeze(images)
            self.squeeze_flag = True

        self.original_shape = images.shape
        self.external_mask = np.zeros(images.shape, dtype=np.int16)
        self.compute_external_mask(input_img=images)
        self.bb_parameters = compute_bounding_box(self.external_mask, padding=2)

        rescaled_input_img, self.final_padding = self.crop_resize_pad(input=images,
                                                                      bb_parameters=self.bb_parameters, image_rows=512,
                                                                      image_cols=512, interpolator='cubic',
                                                                      empty_value='air')
        if self.squeeze_flag:
            rescaled_input_img = np.expand_dims(rescaled_input_img, axis=-1)

        if self.annotation:
            annotation = input_features['annotation']
            if annotation.dtype != 'float32':
                annotation = annotation.astype('float32')
            if self.squeeze_flag:
                annotation = np.squeeze(annotation)
            rescaled_input_label, self.final_padding = self.crop_resize_pad(input=annotation,
                                                                            bb_parameters=self.bb_parameters,
                                                                            image_rows=512,
                                                                            image_cols=512, interpolator='linear_label',
                                                                            threshold=0.5)
            if self.squeeze_flag:
                rescaled_input_label = np.expand_dims(rescaled_input_label, axis=-1)
            input_features['annotation'] = rescaled_input_label
        input_features['image'] = rescaled_input_img
        return input_features

    def post_process(self, input_features):
        images = input_features['image']
        pred = input_features['prediction']

        if self.debug:
            recovered_img = self.recover_original(resize_image=images, original_shape=self.original_shape,
                                                  bb_parameters=self.bb_parameters, final_padding=self.final_padding,
                                                  interpolator='cubic', empty_value='air')
            input_features['image'] = recovered_img

        if self.annotation:
            recovered_annotation = self.recover_original(resize_image=input_features['annotation'],
                                                         original_shape=self.original_shape,
                                                         bb_parameters=self.bb_parameters,
                                                         final_padding=self.final_padding,
                                                         interpolator='linear_label', empty_value='zero')
            input_features['annotation'] = recovered_annotation

        recovered_pred = self.recover_original_hot(resize_image=pred, original_shape=self.original_shape,
                                                   bb_parameters=self.bb_parameters, final_padding=self.final_padding,
                                                   interpolator='cubic_pred', empty_value='zero')
        input_features['prediction'] = recovered_pred
        return input_features

    def compute_external_mask(self, input_img):
        self.external_mask[input_img > self.threshold_value] = self.mask_value
        # self.external_mask = compute_binary_morphology(input_img=self.external_mask, radius=2, morph_type='opening')
        self.external_mask = morphology.opening(image=self.external_mask, selem=morphology.ball(2))
        # self.external_mask = self.keep_main_component(annotations=self.external_mask)
        main_component_filter = Remove_Smallest_Structures()
        self.external_mask = main_component_filter.remove_smallest_component(self.external_mask)

    def recover_original_hot(self, resize_image, original_shape=(191, 512, 512, 13), bb_parameters=[], final_padding=[],
                             interpolator='cubic_label', threshold=0.5, empty_value='min'):
        '''
        # nearest creates a "shifting effect"
        # linear_label works great in general with small dots after recover
        # cubic_label is 100% recovered but more zig-zaggy
        :param resize_image:
        :param original_shape:
        :param bb_parameters:
        :param final_padding:
        :param interpolator:
        :param threshold:
        :param empty_value:
        :return:
        '''
        resized_shape = resize_image.shape
        min_slice, max_slice = final_padding[0][0], resized_shape[0] - final_padding[0][1]
        min_row, max_row = final_padding[1][0], resized_shape[1] - final_padding[1][1]
        min_col, max_col = final_padding[2][0], resized_shape[2] - final_padding[2][1]
        unpadded = resize_image[min_slice:max_slice, min_row:max_row, min_col:max_col]

        bb_shape = (
            bb_parameters[1] - bb_parameters[0], bb_parameters[3] - bb_parameters[2],
            bb_parameters[5] - bb_parameters[4])
        cropped_dimension = np.array(original_shape) - np.array(bb_shape)

        # if unpadded.shape[1] > unpadded.shape[2]:
        #   updt_image_rows = unpadded.shape[1] - cropped_dimension[1]
        #   updt_image_cols = int(round(unpadded.shape[2] * (unpadded.shape[1] - cropped_dimension[1]) / unpadded.shape[1]))
        # elif unpadded.shape[1] < unpadded.shape[2]:
        #   updt_image_rows = int(round(unpadded.shape[1] * (unpadded.shape[2] - cropped_dimension[2]) / unpadded.shape[2]))
        #   updt_image_cols = unpadded.shape[2] - cropped_dimension[2]
        # else:
        #   updt_image_rows = unpadded.shape[1] - cropped_dimension[1]
        #   updt_image_cols = unpadded.shape[2] - cropped_dimension[2]

        if unpadded.shape[1] > unpadded.shape[2]:
            updt_image_rows = original_shape[1] - cropped_dimension[1]
            updt_image_cols = int(
                round(unpadded.shape[2] * (original_shape[1] - cropped_dimension[1]) / unpadded.shape[1]))
        elif unpadded.shape[1] < unpadded.shape[2]:
            updt_image_rows = int(
                round(unpadded.shape[1] * (original_shape[2] - cropped_dimension[2]) / unpadded.shape[2]))
            updt_image_cols = original_shape[2] - cropped_dimension[2]
        else:
            updt_image_rows = original_shape[1] - cropped_dimension[1]
            updt_image_cols = original_shape[2] - cropped_dimension[2]

        target_shape = (unpadded.shape[0], updt_image_rows, updt_image_cols, unpadded.shape[-1])
        rescaled_img = np.zeros(target_shape, dtype=resize_image.dtype)

        if interpolator == 'cubic_label':
            for channel in range(1, unpadded.shape[-1]):
                label = 1
                temp = copy.deepcopy(unpadded[:, :, :, channel])
                temp[temp != label] = 0
                temp[temp > 0] = 1
                for idx in range(unpadded.shape[0]):
                    rescaled_temp = cv2.resize(temp[idx, :, :], (updt_image_cols, updt_image_rows),
                                               interpolation=cv2.INTER_CUBIC)
                    rescaled_temp[rescaled_temp > threshold] = label
                    rescaled_temp[rescaled_temp < label] = 0
                    rescaled_img[idx, :, :, channel][rescaled_temp == label] = label
        if interpolator == 'cubic_pred':
            for channel in range(1, unpadded.shape[-1]):
                temp = copy.deepcopy(unpadded[:, :, :, channel])
                for idx in range(unpadded.shape[0]):
                    rescaled_img[idx, :, :, channel] = cv2.resize(temp[idx, :, :], (updt_image_cols, updt_image_rows),
                                                                  interpolation=cv2.INTER_CUBIC)
        elif interpolator == 'linear_label':
            for channel in range(1, unpadded.shape[-1]):
                label = 1
                temp = copy.deepcopy(unpadded[:, :, :, channel])
                temp[temp != label] = 0
                temp[temp > 0] = 1
                for idx in range(unpadded.shape[0]):
                    rescaled_temp = cv2.resize(temp[idx, :, :], (updt_image_cols, updt_image_rows),
                                               interpolation=cv2.INTER_LINEAR)
                    rescaled_temp[rescaled_temp > threshold] = label
                    rescaled_temp[rescaled_temp < label] = 0
                    rescaled_img[idx, :, :, channel][rescaled_temp == label] = label
        else:
            print('WARNING: No resize performed as the provided interpolator is not compatible')
            print('Supporter interpolator: [cubic_label, linear_label]')

        bb_padding = [[bb_parameters[0], original_shape[0] - bb_parameters[1]],
                      [bb_parameters[2], original_shape[1] - bb_parameters[3]],
                      [bb_parameters[4], original_shape[2] - bb_parameters[5]],
                      [0, 0]]

        if empty_value == 'air':
            recovered_img = np.pad(rescaled_img, bb_padding, 'constant', constant_values=-1000)
        elif empty_value == 'min':
            recovered_img = np.pad(rescaled_img, bb_padding, 'constant', constant_values=np.min(rescaled_img))
        elif empty_value == 'zero':
            recovered_img = np.pad(rescaled_img, bb_padding, 'constant', constant_values=0)
        else:
            recovered_img = np.pad(rescaled_img, bb_padding, 'constant', constant_values=-1000)
        return recovered_img

    def recover_original(self, resize_image, original_shape=(191, 512, 512), bb_parameters=[], final_padding=[],
                         interpolator='linear', threshold=0.5, empty_value='min'):
        '''
        # nearest creates a "shifting effect"
        # linear_label works great in general with small dots after recover
        # cubic_label is 100% recovered but more zig-zaggy
        :param resize_image:
        :param original_shape:
        :param bb_parameters:
        :param final_padding:
        :param interpolator:
        :param threshold:
        :param empty_value:
        :return:
        '''
        resized_shape = resize_image.shape
        min_slice, max_slice = final_padding[0][0], resized_shape[0] - final_padding[0][1]
        min_row, max_row = final_padding[1][0], resized_shape[1] - final_padding[1][1]
        min_col, max_col = final_padding[2][0], resized_shape[2] - final_padding[2][1]
        unpadded = resize_image[min_slice:max_slice, min_row:max_row, min_col:max_col]

        bb_shape = (
            bb_parameters[1] - bb_parameters[0], bb_parameters[3] - bb_parameters[2],
            bb_parameters[5] - bb_parameters[4])
        cropped_dimension = np.array(original_shape) - np.array(bb_shape)

        if unpadded.shape[1] > unpadded.shape[2]:
            updt_image_rows = original_shape[1] - cropped_dimension[1]
            updt_image_cols = int(
                round(unpadded.shape[2] * (original_shape[1] - cropped_dimension[1]) / unpadded.shape[1]))
        elif unpadded.shape[1] < unpadded.shape[2]:
            updt_image_rows = int(
                round(unpadded.shape[1] * (original_shape[2] - cropped_dimension[2]) / unpadded.shape[2]))
            updt_image_cols = original_shape[2] - cropped_dimension[2]
        else:
            updt_image_rows = original_shape[1] - cropped_dimension[1]
            updt_image_cols = original_shape[2] - cropped_dimension[2]

        target_shape = (unpadded.shape[0], updt_image_rows, updt_image_cols)
        rescaled_img = np.zeros(target_shape, dtype=resize_image.dtype)

        if interpolator == 'linear':
            for idx in range(unpadded.shape[0]):
                rescaled_img[idx, :, :] = cv2.resize(unpadded[idx, :, :], (updt_image_cols, updt_image_rows),
                                                     interpolation=cv2.INTER_CUBIC)
        elif interpolator == 'cubic':
            for idx in range(unpadded.shape[0]):
                rescaled_img[idx, :, :] = cv2.resize(unpadded[idx, :, :], (updt_image_cols, updt_image_rows),
                                                     interpolation=cv2.INTER_LINEAR)
        elif interpolator == 'nearest':
            for idx in range(unpadded.shape[0]):
                rescaled_img[idx, :, :] = cv2.resize(unpadded[idx, :, :], (updt_image_cols, updt_image_rows),
                                                     interpolation=cv2.INTER_NEAREST)
        elif interpolator == 'cubic_label':
            for label in range(1, int(unpadded.max()) + 1):
                temp = copy.deepcopy(unpadded)
                temp[temp != label] = 0
                temp[temp > 0] = 1
                for idx in range(unpadded.shape[0]):
                    rescaled_temp = cv2.resize(temp[idx, :, :], (updt_image_cols, updt_image_rows),
                                               interpolation=cv2.INTER_CUBIC)
                    rescaled_temp[rescaled_temp > threshold] = label
                    rescaled_temp[rescaled_temp < label] = 0
                    rescaled_img[idx, :, :][rescaled_temp == label] = label
        elif interpolator == 'linear_label':
            for label in range(1, int(unpadded.max()) + 1):
                temp = copy.deepcopy(unpadded)
                temp[temp != label] = 0
                temp[temp > 0] = 1
                for idx in range(unpadded.shape[0]):
                    rescaled_temp = cv2.resize(temp[idx, :, :], (updt_image_cols, updt_image_rows),
                                               interpolation=cv2.INTER_LINEAR)
                    rescaled_temp[rescaled_temp > threshold] = label
                    rescaled_temp[rescaled_temp < label] = 0
                    rescaled_img[idx, :, :][rescaled_temp == label] = label
        else:
            print('WARNING: No resize performed as the provided interpolator is not compatible')
            print('Supporter interpolator: [linear, cubic, nearest]')

        bb_padding = [[bb_parameters[0], original_shape[0] - bb_parameters[1]],
                      [bb_parameters[2], original_shape[1] - bb_parameters[3]],
                      [bb_parameters[4], original_shape[2] - bb_parameters[5]]]

        if empty_value == 'air':
            recovered_img = np.pad(rescaled_img, bb_padding, 'constant', constant_values=-1000)
        elif empty_value == 'min':
            recovered_img = np.pad(rescaled_img, bb_padding, 'constant', constant_values=np.min(rescaled_img))
        elif empty_value == 'zero':
            recovered_img = np.pad(rescaled_img, bb_padding, 'constant', constant_values=0)
        else:
            recovered_img = np.pad(rescaled_img, bb_padding, 'constant', constant_values=-1000)
        return recovered_img

    def crop_resize_pad(self, input, bb_parameters=[], image_rows=512, image_cols=512, interpolator='linear',
                        threshold=0.5, empty_value='min'):
        '''
        # nearest creates a "shifting effect"
        # linear_label works great in general with small dots after recover
        # cubic_label is 100% recovered but more zig-zaggy
        :param input:
        :param bb_parameters:
        :param image_rows:
        :param image_cols:
        :param interpolator:
        :param threshold:
        :param empty_value:
        :return:
        '''

        input = input[bb_parameters[0]:bb_parameters[1], bb_parameters[2]:bb_parameters[3],
                bb_parameters[4]:bb_parameters[5]]

        if input.shape[1] > input.shape[2]:
            updt_image_rows = image_rows
            updt_image_cols = int(round(input.shape[2] * image_rows / input.shape[1]))
        elif input.shape[1] < input.shape[2]:
            updt_image_rows = int(round(input.shape[1] * image_cols / input.shape[2]))
            updt_image_cols = image_cols
        else:
            updt_image_rows = image_rows
            updt_image_cols = image_cols

        print('   Update rows, cols from {}, {} to {}, {}'.format(input.shape[1], input.shape[2], updt_image_rows,
                                                                  updt_image_cols))

        resized_img = np.zeros((input.shape[0], updt_image_rows, updt_image_cols), dtype=input.dtype)

        if interpolator == 'linear':
            for idx in range(input.shape[0]):
                resized_img[idx, :, :] = cv2.resize(input[idx, :, :], (updt_image_cols, updt_image_rows),
                                                    interpolation=cv2.INTER_LINEAR)
        elif interpolator == 'cubic':
            for idx in range(input.shape[0]):
                resized_img[idx, :, :] = cv2.resize(input[idx, :, :], (updt_image_cols, updt_image_rows),
                                                    interpolation=cv2.INTER_CUBIC)
        elif interpolator == 'nearest':
            for idx in range(input.shape[0]):
                resized_img[idx, :, :] = cv2.resize(input[idx, :, :], (updt_image_cols, updt_image_rows),
                                                    interpolation=cv2.INTER_NEAREST)
        elif interpolator == 'cubic_label':
            for label in range(1, int(input.max()) + 1):
                temp = copy.deepcopy(input)
                temp[temp != label] = 0
                temp[temp > 0] = 1
                for idx in range(input.shape[0]):
                    resized_temp = cv2.resize(temp[idx, :, :], (updt_image_cols, updt_image_rows),
                                              interpolation=cv2.INTER_CUBIC)
                    resized_temp[resized_temp > threshold] = label
                    resized_temp[resized_temp < label] = 0
                    resized_img[idx, :, :][resized_temp == label] = label
        elif interpolator == 'linear_label':
            for label in range(1, int(input.max()) + 1):
                temp = copy.deepcopy(input)
                temp[temp != label] = 0
                temp[temp > 0] = 1
                for idx in range(input.shape[0]):
                    resized_temp = cv2.resize(temp[idx, :, :], (updt_image_cols, updt_image_rows),
                                              interpolation=cv2.INTER_LINEAR)
                    resized_temp[resized_temp > threshold] = label
                    resized_temp[resized_temp < label] = 0
                    resized_img[idx, :, :][resized_temp == label] = label
        else:
            print('WARNING: No resize performed as the provided interpolator is not compatible')
            print('Supporter interpolator: [linear, cubic, nearest]')

        if empty_value == 'air':
            constant_values = -1000
        elif empty_value == 'min':
            constant_values = np.min(input)
        elif empty_value == 'zero':
            constant_values = 0
        else:
            constant_values = -1000

        # two steps padding to avoid odd padding
        holder = [resized_img.shape[0], image_rows, image_cols] - np.asarray(resized_img.shape)
        final_padding = [[max([int(i / 2), 0]), max([int(i / 2), 0])] for i in holder]
        resized_img = np.pad(resized_img, final_padding, 'constant', constant_values=constant_values)

        holder = [resized_img.shape[0], image_rows, image_cols] - np.asarray(resized_img.shape)
        supp_padding = [[0, i] for i in holder]
        if np.max(supp_padding) > 0:
            resized_img = np.pad(resized_img, supp_padding, 'constant', constant_values=constant_values)
            final_padding = list(np.array(final_padding) + np.array(supp_padding))
        return resized_img, final_padding

    def keep_main_component(self, annotations):
        labels = morphology.label(annotations, connectivity=3)
        if np.max(labels) > 1:
            area = []
            max_val = 0
            for i in range(1, labels.max() + 1):
                new_area = labels[labels == i].shape[0]
                area.append(new_area)
                if new_area == max(area):
                    max_val = i
            labels[labels != max_val] = 0
            labels[labels > 0] = 1
            annotations = labels
        return annotations


class CreateUpperVagina(ImageProcessor):
    def __init__(self, prediction_keys=('prediction',), class_id=(5,), sup_margin=(20,)):
        self.prediction_keys = prediction_keys
        self.class_id = class_id
        self.sup_margin = sup_margin

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.prediction_keys)
        for key, class_id, sup_margin in zip(self.prediction_keys, self.class_id, self.sup_margin):
            prediction = input_features[key]
            # test if prediction class id is empty
            if np.any(prediction[..., class_id]):
                min_slice, max_slice, min_row, max_row, min_col, max_col = compute_bounding_box(
                    prediction[..., class_id],
                    padding=0)
                spacing = input_features['spacing']
                nb_slices = ceil(sup_margin / spacing[-1])
                new_prediction = np.zeros(prediction.shape[0:-1] + (prediction.shape[-1] + 1,), dtype=prediction.dtype)
                new_prediction[..., 0:prediction.shape[-1]] = prediction
                new_prediction[..., -1][(max_slice + 1 - nb_slices):(max_slice + 1), ...] = \
                    new_prediction[..., class_id][(max_slice + 1 - nb_slices):(max_slice + 1), ...]
            else:
                # create empty class
                new_prediction = np.zeros(prediction.shape[0:-1] + (prediction.shape[-1] + 1,), dtype=prediction.dtype)
                new_prediction[..., 0:prediction.shape[-1]] = prediction

            input_features[key] = new_prediction
        return input_features


class CombinePredictions(ImageProcessor):
    def __init__(self, prediction_keys=('prediction',), combine_ids=((7, 8),), closings=(False,)):
        self.prediction_keys = prediction_keys
        self.combine_ids = combine_ids
        self.closings = closings

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.prediction_keys)
        for key, combine_id, closing in zip(self.prediction_keys, self.combine_ids, self.closings):
            prediction = input_features[key]
            new_prediction = np.zeros(prediction.shape[0:-1] + (prediction.shape[-1] + 1,), dtype=prediction.dtype)
            new_prediction[..., 0:prediction.shape[-1]] = prediction
            # combine ID into the last class, this is faster than doing prediction.sum(where=mask)
            for id in combine_id:
                new_prediction[..., -1] += prediction[..., id]
            # threshold to remove overlap
            new_prediction[..., -1][new_prediction[..., -1] > 0] = 1
            if closing:
                new_prediction[..., -1] = compute_binary_morphology(new_prediction[..., -1], radius=2,
                                                                    morph_type='closing')
            input_features[key] = new_prediction

        return input_features


def expand_box_indexes(z_start, z_stop, r_start, r_stop, c_start, c_stop, annotation_shape, bounding_box_expansion):
    if len(bounding_box_expansion) == 3:
        z_start = max([0, z_start - floor(bounding_box_expansion[0] / 2)])
        z_stop = min([annotation_shape[0], z_stop + ceil(bounding_box_expansion[0] / 2)])
        r_start = max([0, r_start - floor(bounding_box_expansion[1] / 2)])
        r_stop = min([annotation_shape[1], r_stop + ceil(bounding_box_expansion[1] / 2)])
        c_start = max([0, c_start - floor(bounding_box_expansion[2] / 2)])
        c_stop = min([annotation_shape[2], c_stop + ceil(bounding_box_expansion[2] / 2)])
    elif len(bounding_box_expansion) == 6:
        z_start = max([0, z_start - int(bounding_box_expansion[0])])
        z_stop = min([annotation_shape[0], z_stop + int(bounding_box_expansion[1])])
        r_start = max([0, r_start - int(bounding_box_expansion[2])])
        r_stop = min([annotation_shape[1], r_stop + int(bounding_box_expansion[3])])
        c_start = max([0, c_start - int(bounding_box_expansion[4])])
        c_stop = min([annotation_shape[2], c_stop + int(bounding_box_expansion[5])])
    else:
        raise ValueError("bounding_box_expansion should be a tuple of len 3 or 6")
    return z_start, z_stop, r_start, r_stop, c_start, c_stop


def get_bounding_boxes(annotation_handle, value, extract_comp=True):
    stats = sitk.LabelShapeStatisticsImageFilter()
    if extract_comp:
        Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
        RelabelComponent = sitk.RelabelComponentImageFilter()
        RelabelComponent.SortByObjectSizeOn()
        thresholded_image = sitk.BinaryThreshold(annotation_handle, lowerThreshold=value, upperThreshold=value + 1)
        connected_image = Connected_Component_Filter.Execute(thresholded_image)
        connected_image = RelabelComponent.Execute(connected_image)
        stats.Execute(connected_image)
    else:
        Cast_Image_Filter = sitk.CastImageFilter()
        Cast_Image_Filter.SetNumberOfThreads(0)
        Cast_Image_Filter.SetOutputPixelType(sitk.sitkUInt32)
        casted_image = Cast_Image_Filter.Execute(annotation_handle)
        stats.Execute(casted_image)
    bounding_boxes = [stats.GetBoundingBox(l) for l in stats.GetLabels()]
    num_voxels = np.asarray([stats.GetNumberOfPixels(l) for l in stats.GetLabels()]).astype('float32')
    return bounding_boxes, num_voxels


def add_bounding_box_to_dict(bounding_box, input_features=None, val=None, return_indexes=False,
                             add_to_dictionary=False):
    c_start, r_start, z_start, c_stop, r_stop, z_stop = bounding_box
    z_stop, r_stop, c_stop = z_start + z_stop, r_start + r_stop, c_start + c_stop
    if return_indexes:
        return z_start, z_stop, r_start, r_stop, c_start, c_stop
    if add_to_dictionary:
        input_features['bounding_boxes_z_start_{}'.format(val)] = z_start
        input_features['bounding_boxes_r_start_{}'.format(val)] = r_start
        input_features['bounding_boxes_c_start_{}'.format(val)] = c_start
        input_features['bounding_boxes_z_stop_{}'.format(val)] = z_stop
        input_features['bounding_boxes_r_stop_{}'.format(val)] = r_stop
        input_features['bounding_boxes_c_stop_{}'.format(val)] = c_stop
    return input_features


class Add_Bounding_Box_Indexes(ImageProcessor):
    def __init__(self, wanted_vals_for_bbox=None, add_to_dictionary=False, label_name='annotation', extract_comp=True):
        '''
        :param wanted_vals_for_bbox: a list of values in integer form for bboxes
        '''
        assert type(wanted_vals_for_bbox) is list, 'Provide a list for bboxes'
        self.wanted_vals_for_bbox = wanted_vals_for_bbox
        self.add_to_dictionary = add_to_dictionary
        self.label_name = label_name
        self.extract_comp = extract_comp

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.label_name)
        annotation_base = input_features[self.label_name]
        for val in self.wanted_vals_for_bbox:
            temp_val = val
            if len(annotation_base.shape) > 3:
                annotation = (annotation_base[..., val] > 0).astype('int')
                temp_val = 1
            else:
                annotation = annotation_base
            slices = np.where(annotation == temp_val)
            if slices:
                bounding_boxes, voxel_volumes = get_bounding_boxes(sitk.GetImageFromArray(annotation), temp_val,
                                                                   extract_comp=self.extract_comp)
                input_features['voxel_volumes_{}'.format(val)] = voxel_volumes
                input_features['bounding_boxes_{}'.format(val)] = bounding_boxes
                input_features = add_bounding_box_to_dict(input_features=input_features, bounding_box=bounding_boxes[0],
                                                          val=val, return_indexes=False,
                                                          add_to_dictionary=self.add_to_dictionary)
        return input_features


class Clip_Images_By_Extension(ImageProcessor):
    def __init__(self, input_keys=('image',), annotation_keys=('annotation',), post_process_keys=('prediction',),
                 inf_extension=np.inf, sup_extension=np.inf, use_spacing=False,
                 spacing_key='spacing'):
        '''
        input_keys: input keys to loop over
        annotation_keys: annotation keys to define the inf/sup extension on each image
        inf_extension: inferior extension (mm)
        sup_extension: superior extension (mm)
        use_spacing: flag if use spacing to convert mm to number of slice
        spacing_handle_key: (if use_spacing==True) key to get the spacing list from the input_features dictionary
        '''
        self.input_keys = input_keys
        self.annotation_keys = annotation_keys
        self.inf_extension = inf_extension
        self.sup_extension = sup_extension
        self.use_spacing = use_spacing
        self.spacing_key = spacing_key
        self.post_process_keys = post_process_keys

    def get_start_stop(self, annotation, inf_extension=np.inf, sup_extension=np.inf, desired_val=1):
        if len(annotation.shape) > 3:
            annotation = np.argmax(annotation, axis=-1)
        non_zero_values = np.where(np.max(annotation, axis=(1, 2)) >= desired_val)[0]
        start, stop = -1, -1
        if non_zero_values.any():
            start = int(non_zero_values[0])
            stop = int(non_zero_values[-1])
            start = max([start - inf_extension, 0])
            stop = min([stop + sup_extension, annotation.shape[0]])
        return start, stop

    def pre_process(self, input_features):

        if self.use_spacing:
            inf_extension = floor(self.inf_extension / input_features[self.spacing_key][-1])
            sup_extension = floor(self.sup_extension / input_features[self.spacing_key][-1])
        else:
            inf_extension = self.inf_extension
            sup_extension = self.sup_extension

        for image_key, annotation_key in zip(self.input_keys, self.annotation_keys):
            image = input_features[image_key]
            annotation = input_features[annotation_key]
            start, stop = self.get_start_stop(annotation, inf_extension, sup_extension)
            input_features['og_shape'] = image.shape
            input_features['og_shape_{}'.format(image_key)] = image.shape
            input_features['start'] = start
            input_features['stop'] = stop
            if start != -1 and stop != -1:
                image, annotation = image[start:stop, ...], annotation[start:stop, ...]
            input_features[image_key] = image
            input_features[annotation_key] = annotation.astype('int8')
        return input_features

    def post_process(self, input_features):
        for image_key in self.post_process_keys:
            og_shape = input_features['og_shape']
            image = input_features[image_key]
            start = input_features['start']
            stop = input_features['stop']
            pads = [(start, og_shape[0] - stop), (0, 0), (0, 0)]
            if len(image.shape) > 3:
                pads += [(0, 0)]
            image = np.pad(image, pads, constant_values=np.min(image))
            input_features[image_key] = image
        return input_features


class sITK_Handle_to_Numpy(ImageProcessor):
    def __init__(self, image_keys=('image',), post_process_keys=('prediction',)):
        self.image_keys = image_keys
        self.post_process_keys = post_process_keys

    def pre_process(self, input_features):
        for image_key in self.image_keys:
            handle = input_features[image_key]
            if not isinstance(handle, np.ndarray):
                numpy_array = sitk.GetArrayFromImage(handle)
                input_features[image_key] = numpy_array
        return input_features

    def post_process(self, input_features):
        for image_key in self.post_process_keys:
            handle = input_features[image_key]
            if not isinstance(handle, np.ndarray):
                numpy_array = sitk.GetArrayFromImage(handle)
                input_features[image_key] = numpy_array
        return input_features


class Box_Images(ImageProcessor):
    def __init__(self, bounding_box_expansion, image_keys=('image',), annotation_key='annotation',
                 wanted_vals_for_bbox=None,
                 power_val_z=1, power_val_r=1, power_val_c=1, min_images=None, min_rows=None, min_cols=None,
                 post_process_keys=('image', 'annotation', 'prediction'), pad_value=None, extract_comp=True):
        """
        :param image_keys: keys which corresponds to an image to be normalized
        :param annotation_key: key which corresponds to an annotation image used for normalization
        :param wanted_vals_for_bbox:
        :param bounding_box_expansion:
        :param power_val_z:
        :param power_val_r:
        :param power_val_c:
        :param min_images:
        :param min_rows:
        :param min_cols:
        """
        assert type(wanted_vals_for_bbox) in [list, tuple], 'Provide a list for bboxes'
        self.wanted_vals_for_bbox = wanted_vals_for_bbox
        self.bounding_box_expansion = bounding_box_expansion
        self.power_val_z, self.power_val_r, self.power_val_c = power_val_z, power_val_r, power_val_c
        self.min_images, self.min_rows, self.min_cols = min_images, min_rows, min_cols
        self.image_keys, self.annotation_key = image_keys, annotation_key
        self.post_process_keys = post_process_keys
        self.pad_value = pad_value
        self.extract_comp = extract_comp

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.image_keys + (self.annotation_key,))
        annotation = input_features[self.annotation_key]
        if len(annotation.shape) > 3:
            mask = np.zeros(annotation.shape[:-1])
            argmax_annotation = np.argmax(annotation, axis=-1)
            for val in self.wanted_vals_for_bbox:
                mask[argmax_annotation == val] = 1
        else:
            mask = np.zeros(annotation.shape)
            for val in self.wanted_vals_for_bbox:
                mask[annotation == val] = 1
        for val in [1]:
            add_indexes = Add_Bounding_Box_Indexes([val], label_name='mask', extract_comp=self.extract_comp)
            input_features['mask'] = mask
            add_indexes.pre_process(input_features)
            del input_features['mask']
            z_start, z_stop, r_start, r_stop, c_start, c_stop = add_bounding_box_to_dict(
                input_features['bounding_boxes_{}'.format(val)][0], return_indexes=True)

            z_start, z_stop, r_start, r_stop, c_start, c_stop = expand_box_indexes(z_start, z_stop, r_start, r_stop,
                                                                                   c_start, c_stop,
                                                                                   annotation_shape=annotation.shape,
                                                                                   bounding_box_expansion=
                                                                                   self.bounding_box_expansion)

            z_total, r_total, c_total = z_stop - z_start, r_stop - r_start, c_stop - c_start
            remainder_z, remainder_r, remainder_c = self.power_val_z - z_total % self.power_val_z if z_total % self.power_val_z != 0 else 0, \
                                                    self.power_val_r - r_total % self.power_val_r if r_total % self.power_val_r != 0 else 0, \
                                                    self.power_val_c - c_total % self.power_val_c if c_total % self.power_val_c != 0 else 0
            remainders = np.asarray([remainder_z, remainder_r, remainder_c])
            z_start, z_stop, r_start, r_stop, c_start, c_stop = expand_box_indexes(z_start, z_stop, r_start, r_stop,
                                                                                   c_start, c_stop,
                                                                                   annotation_shape=
                                                                                   annotation.shape,
                                                                                   bounding_box_expansion=
                                                                                   remainders)
            min_images, min_rows, min_cols = z_total + remainder_z, r_total + remainder_r, c_total + remainder_c
            remainders = [0, 0, 0]
            if self.min_images is not None:
                remainders[0] = max([0, self.min_images - min_images])
                min_images = max([min_images, self.min_images])
            if self.min_rows is not None:
                remainders[1] = max([0, self.min_rows - min_rows])
                min_rows = max([min_rows, self.min_rows])
            if self.min_cols is not None:
                remainders[2] = max([0, self.min_cols - min_cols])
                min_cols = max([min_cols, self.min_cols])
            remainders = np.asarray(remainders)
            z_start, z_stop, r_start, r_stop, c_start, c_stop = expand_box_indexes(z_start, z_stop, r_start, r_stop,
                                                                                   c_start, c_stop,
                                                                                   annotation_shape=
                                                                                   annotation.shape,
                                                                                   bounding_box_expansion=
                                                                                   remainders)
            input_features['z_r_c_start'] = [z_start, r_start, c_start]
            for key in self.image_keys:
                image = input_features[key]
                input_features['og_shape'] = image.shape
                input_features['og_shape_{}'.format(key)] = image.shape
                image_cube = image[z_start:z_stop, r_start:r_stop, c_start:c_stop]
                img_shape = image_cube.shape
                pads = [min_images - img_shape[0], min_rows - img_shape[1], min_cols - img_shape[2]]
                pads = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
                if self.pad_value is not None:
                    pad_value = self.pad_value
                else:
                    pad_value = np.min(image_cube)
                while len(image_cube.shape) > len(pads):
                    pads += [[0, 0]]
                image_cube = np.pad(image_cube, pads, constant_values=pad_value)
                input_features[key] = image_cube.astype(image.dtype)
                input_features['pads'] = [pads[i][0] for i in range(3)]
            annotation_cube = annotation[z_start:z_stop, r_start:r_stop, c_start:c_stop]
            pads = [min_images - annotation_cube.shape[0], min_rows - annotation_cube.shape[1],
                    min_cols - annotation_cube.shape[2]]
            if len(annotation.shape) > 3:
                pads = np.append(pads, [0])
            pads = [[max([0, floor(i / 2)]), max([0, ceil(i / 2)])] for i in pads]
            annotation_cube = np.pad(annotation_cube, pads)
            if len(annotation.shape) > 3:
                annotation_cube[..., 0] = 1 - np.sum(annotation_cube[..., 1:], axis=-1)
            input_features[self.annotation_key] = annotation_cube.astype(annotation.dtype)
        return input_features

    def post_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.post_process_keys)
        for key in self.post_process_keys:
            image = input_features[key]
            pads = input_features['pads']
            image = image[pads[0]:, pads[1]:, pads[2]:, ...]
            pads = [(i, 0) for i in input_features['z_r_c_start']]
            while len(image.shape) > len(pads):
                pads += [(0, 0)]
            image = np.pad(image, pads, constant_values=np.min(image))
            og_shape = input_features['og_shape']
            im_shape = image.shape
            if im_shape[0] > og_shape[0]:
                dif = og_shape[0] - im_shape[0]
                image = image[:dif]
            if im_shape[1] > og_shape[1]:
                dif = og_shape[1] - im_shape[1]
                image = image[:, :dif]
            if im_shape[2] > og_shape[2]:
                dif = og_shape[2] - im_shape[2]
                image = image[:, :, :dif]
            im_shape = image.shape
            pads = [(0, og_shape[0] - im_shape[0]), (0, og_shape[1] - im_shape[1]), (0, og_shape[2] - im_shape[2])]
            if len(image.shape) > 3:
                pads += [(0, 0)]
            image = np.pad(image, pads, constant_values=np.min(image))
            input_features[key] = image
        return input_features


class Duplicate_Prediction(ImageProcessor):
    def __init__(self, prediction_key='prediction'):
        self.prediction_key = prediction_key

    def pre_process(self, input_features):
        prediction = input_features[self.prediction_key]
        if prediction.shape[-1] != 2:
            assert ValueError("Prediction shape should be 2 for binary class prediction (background and output class)")

        new_prediction = np.zeros(prediction.shape[0:-1] + (prediction.shape[-1] + 1,), dtype=prediction.dtype)
        # copy current prediction to new array
        new_prediction[..., 0:prediction.shape[-1]] = prediction
        # duplicate last dim
        new_prediction[..., -1] = prediction[..., -1]
        input_features[self.prediction_key] = new_prediction
        return input_features


class ZNorm_By_Annotation(ImageProcessor):
    def __init__(self, image_key='image', annotation_key='annotation'):
        self.image_key = image_key
        self.annotation_key = annotation_key

    def pre_process(self, input_features):
        input_features['mean'] = np.mean(input_features[self.image_key][input_features[self.annotation_key] > 0])
        input_features['std'] = np.std(input_features[self.image_key][input_features[self.annotation_key] > 0])
        mean_val = input_features['mean']
        std_val = input_features['std']
        image = input_features[self.image_key]
        image = (image - mean_val) / std_val
        input_features[self.image_key] = image
        return input_features

    def post_process(self, input_features):
        return input_features


class Normalize_Images(ImageProcessor):
    def __init__(self, keys=('image',), mean_values=(0,), std_values=(1,), ):
        """
        :param keys: tuple of image keys
        :param mean_values: tuple of mean values
        :param std_values: tuple of standard deviations
        """
        self.keys = keys
        self.mean_values = mean_values
        self.std_values = std_values

    def parse(self, image_features):
        _check_keys_(image_features, self.keys)
        for key, mean_val, std_val in zip(self.keys, self.mean_values, self.std_values):
            mean_val = tf.constant(mean_val, dtype=image_features[key].dtype)
            std_val = tf.constant(std_val, dtype=image_features[key].dtype)
            image_features[key] = (image_features[key] - mean_val) / std_val
        return image_features

    def pre_process(self, input_features):
        _check_keys_(input_features, self.keys)
        for key, mean_val, std_val in zip(self.keys, self.mean_values, self.std_values):
            image = input_features[key]
            image = (image - mean_val) / std_val
            input_features[key] = image
        return input_features

    def post_process(self, input_features):
        return input_features


class Threshold_Images(ImageProcessor):
    def __init__(self, image_keys=('image',), lower_bounds=(-np.inf,), upper_bounds=(np.inf,), divides=(False,)):
        """
        :param keys: tuple of image keys
        :param lower_bounds: tuple of bounds
        :param upper_bounds: tuple of bounds
        :param divides: boolean if you want to divide
        """
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.image_keys = image_keys
        self.divides = divides

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(image_features, self.image_keys)
        for key, lower_bound, upper_bound, divide in zip(self.image_keys, self.lower_bounds, self.upper_bounds,
                                                         self.divides):
            image_features[key] = tf.where(image_features[key] > tf.cast(upper_bound, dtype=image_features[key].dtype),
                                           tf.cast(upper_bound, dtype=image_features[key].dtype), image_features[key])
            image_features[key] = tf.where(image_features[key] < tf.cast(lower_bound, dtype=image_features[key].dtype),
                                           tf.cast(lower_bound, dtype=image_features[key].dtype), image_features[key])
            if divide:
                image_features[key] = tf.divide(image_features[key], tf.cast(tf.subtract(upper_bound, lower_bound),
                                                                             dtype=image_features[key].dtype))
        return image_features

    def pre_process(self, input_features):
        _check_keys_(input_features, self.image_keys)
        for key, lower_bound, upper_bound, divide in zip(self.image_keys, self.lower_bounds, self.upper_bounds,
                                                         self.divides):
            image = input_features[key]
            image[image < lower_bound] = lower_bound
            image[image > upper_bound] = upper_bound
            if divide:
                image = image / (upper_bound - lower_bound)
            input_features[key] = image

        return input_features

    def post_process(self, input_features):
        return input_features


class Postprocess_Pancreas(ImageProcessor):
    def __init__(self, max_comp=2, dist=95, radius=1, prediction_keys=('prediction',)):
        self.max_comp = max_comp
        self.dist = dist
        self.radius = radius
        self.prediction_keys = prediction_keys

    def compute_centroid(self, annotation):
        '''
        :param annotation: A binary image of shape [# images, # rows, # cols, channels]
        :return: index of centroid
        '''
        shape = annotation.shape
        indexes = np.where(np.any(annotation, axis=(1, 2)) == True)[0]
        index_slice = int(np.mean(indexes))
        indexes = np.where(np.any(annotation, axis=(0, 2)) == True)[0]
        index_row = int(np.mean(indexes))
        indexes = np.where(np.any(annotation, axis=(0, 1)) == True)[0]
        index_col = int(np.mean(indexes))
        return (index_slice, index_row, index_col)

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.prediction_keys)
        for key in self.prediction_keys:
            pred = input_features[key]
            for index in range(1, pred.shape[-1]):
                temp_pred = pred[..., index]
                opened_pred = compute_binary_morphology(input_img=temp_pred, radius=1, morph_type='closing')
                pred[..., index] = extract_main_component(nparray=opened_pred, dist=95, max_comp=2)

        return input_features


class Remove_Annotations(ImageProcessor):
    def __init__(self, keep_annotation_id=[1, 2, 3, 4, 5, 6]):
        self.keep_annotation_id = keep_annotation_id

    def pre_process(self, input_features):

        if len(input_features['annotation'].shape) == 3:
            annotation_handle = input_features['annotation']
            output = np.zeros(annotation_handle.shape, annotation_handle.dtype)
            new_id = 1
            for i in range(1, np.max(annotation_handle) + 1):
                if i not in self.keep_annotation_id:
                    continue
                output[annotation_handle == i] = new_id
                new_id += 1
            input_features['annotation'] = output
        return input_features


class Combine_Annotations_To_Mask(ImageProcessor):
    def __init__(self, annotation_input=[1, 2], to_annotation=1, annotation_key='annotation', output_key='mask'):
        self.annotation_input = annotation_input
        self.to_annotation = to_annotation
        self.annotation_key = annotation_key
        self.output_key = output_key

    def pre_process(self, input_features):
        annotation = copy.deepcopy(input_features[self.annotation_key])
        assert len(annotation.shape) == 3 or len(
            annotation.shape) == 4, 'To combine annotations the size has to be 3 or 4'
        if len(annotation.shape) == 3:
            for val in self.annotation_input:
                annotation[annotation == val] = self.to_annotation
        elif len(annotation.shape) == 4:
            annotation[..., self.to_annotation] += annotation[..., self.annotation_input]
            del annotation[..., self.annotation_input]
        input_features[self.output_key] = annotation
        return input_features


class Mask_Image(ImageProcessor):
    def __init__(self, masked_value=0):
        self.masked_value = masked_value

    def parse(self, image_features):
        mask = image_features['mask']
        mask = tf.expand_dims(mask, axis=-1)
        image_features['image'] = tf.where(mask == 0, tf.cast(self.masked_value, dtype=image_features['image'].dtype),
                                           image_features['image'])
        return image_features


class Per_Patient_ZNorm(ImageProcessor):

    def __init__(self, image_keys=('image',), dtypes=('float16',), lower_bound=-3.55, upper_bound=3.55):
        self.image_keys = image_keys
        self.dtypes = dtypes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def pre_process(self, input_features):
        input_features['mean'] = np.mean(input_features['image'][input_features['annotation'] > 0])
        input_features['std'] = np.std(input_features['image'][input_features['annotation'] > 0])
        return input_features

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys + ('mean', 'std',))
        for key, dtype in zip(self.image_keys, self.dtypes):
            image = tf.cast(input_features[key], dtype='float32')
            image = tf.math.divide(
                tf.math.subtract(
                    image,
                    input_features['mean']
                ),
                input_features['std']
            )
            image = tf.where(image > tf.cast(self.upper_bound, dtype=image.dtype),
                             tf.cast(self.upper_bound, dtype=image.dtype), image)
            image = tf.where(image < tf.cast(self.lower_bound, dtype=image.dtype),
                             tf.cast(self.lower_bound, dtype=image.dtype), image)
            input_features[key] = tf.cast(image, dtype=dtype)
        return input_features


class Expand_Dimensions_Per_Key(ImageProcessor):
    def __init__(self, axis=-1, image_keys=('image',), ):
        self.axis = axis
        self.image_keys = image_keys

    def parse(self, image_features, *args, **kwargs):
        for key in self.image_keys:
            image_features[key] = tf.expand_dims(image_features[key], axis=self.axis)
        return image_features


class Repeat_Channel_Per_Key(ImageProcessor):
    def __init__(self, axis=-1, repeats=3, image_keys=('image',), ):
        '''
        :param axis: axis to expand
        :param repeats: number of repeats
        '''
        self.axis = axis
        self.repeats = repeats
        self.image_keys = image_keys

    def parse(self, input_features, *args, **kwargs):
        for key in self.image_keys:
            input_features[key] = tf.repeat(input_features[key], axis=self.axis, repeats=self.repeats)

        return input_features


class Ensure_Image_Key_Proportions(ImageProcessor):
    def __init__(self, image_rows=512, image_cols=512, preserve_aspect_ratio=False, image_keys=('image', 'annotation'),
                 interp=('bilinear', 'nearest')):
        self.image_rows = tf.constant(image_rows)
        self.image_cols = tf.constant(image_cols)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.image_keys = image_keys
        self.interp = interp

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)

        for key, method in zip(self.image_keys, self.interp):
            assert len(input_features[key].shape) > 2, 'You should do an expand_dimensions before this!'

            input_features[key] = tf.image.resize(input_features[key], (self.image_rows, self.image_cols),
                                                  method=method, preserve_aspect_ratio=self.preserve_aspect_ratio)
            input_features[key] = tf.image.resize_with_crop_or_pad(input_features[key],
                                                                   target_width=self.image_cols,
                                                                   target_height=self.image_rows)
        return input_features


class Per_Image_Z_Normalization(ImageProcessor):
    def __init__(self, image_keys=('image',), dtypes=('float16',)):
        '''
        :param image_keys:
        '''
        self.image_keys = image_keys
        self.dtypes = dtypes

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key, dtype in zip(self.image_keys, self.dtypes):
            image = tf.cast(input_features[key], dtype='float32')
            image = tf.image.per_image_standardization(image=image)
            input_features[key] = tf.cast(image, dtype=dtype)

        return input_features


class ImageHistogramEqualizer(ImageProcessor):
    def __init__(self, image_keys=('image',), dtypes=('float16',), norm_flag=False, recover=False):
        '''
        :param image_keys:
        '''
        self.image_keys = image_keys
        self.dtypes = dtypes
        self.norm_flag = norm_flag
        self.recover = recover

    def min_max_normalization(self, image):
        image = tf.math.divide(
            tf.math.subtract(
                image,
                tf.reduce_min(image)
            ),
            tf.math.subtract(
                tf.reduce_max(image),
                tf.reduce_min(image)
            )
        )
        image = tf.multiply(image, tf.cast(255, image.dtype))
        return image

    def recover_normalization(self, image, min, max):
        image = tf.divide(image, tf.cast(255, image.dtype))
        image = tf.math.multiply(
            image,
            tf.math.subtract(
                max,
                min
            ),
        )
        image = tf.math.add(
            image,
            min
        )
        return image

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key, dtype in zip(self.image_keys, self.dtypes):
            image = tf.cast(input_features[key], dtype='float32')
            if self.recover:
                min, max = tf.reduce_min(image), tf.reduce_max(image)
            if self.norm_flag:
                image = self.min_max_normalization(image)
            image = tfa.image.equalize(image=image)
            if self.recover:
                image = self.recover_normalization(image, min, max)
            input_features[key] = tf.cast(image, dtype=dtype)
        return input_features


class DilateBinary(ImageProcessor):
    def __init__(self, image_keys=('annotation',), radius=(5,), run_post_process=False):
        self.image_keys = image_keys
        self.radius = radius
        self.run_post_process = run_post_process

    def pre_process(self, input_features):
        _check_keys_(input_features, self.image_keys)
        for key, radius in zip(self.image_keys, self.radius):
            image = input_features[key]
            dtype = image.dtype
            image_handle = sitk.GetImageFromArray(image.astype(dtype=np.uint8))
            binary_dilate_filter = sitk.BinaryDilateImageFilter()
            binary_dilate_filter.SetNumberOfThreads(0)
            binary_dilate_filter.SetKernelRadius(radius)
            image_handle = binary_dilate_filter.Execute(image_handle)
            image = sitk.GetArrayFromImage(image_handle).astype(dtype=dtype)
            input_features[key] = image
        return input_features

    def post_process(self, input_features):
        if not self.run_post_process:
            return input_features

        _check_keys_(input_features, self.image_keys)
        for key, radius in zip(self.image_keys, self.radius):
            image = input_features[key]
            dtype = image.dtype
            image_handle = sitk.GetImageFromArray(image.astype(dtype=np.uint8))
            binary_erode_filter = sitk.BinaryErodeImageFilter()
            binary_erode_filter.SetNumberOfThreads(0)
            binary_erode_filter.SetKernelRadius(radius)
            image_handle = binary_erode_filter.Execute(image_handle)
            image = sitk.GetArrayFromImage(image_handle).astype(dtype=dtype)
            input_features[key] = image
        return input_features


class Per_Image_MinMax_Normalization(ImageProcessor):
    def __init__(self, image_keys=('image',), threshold_value=255.0):
        '''
        :param image_keys:
        :param treshold_value:
        '''
        self.image_keys = image_keys
        self.threshold_value = threshold_value

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key in self.image_keys:
            image = input_features[key]
            image = tf.math.divide(
                tf.math.subtract(
                    image,
                    tf.reduce_min(image)
                ),
                tf.math.subtract(
                    tf.reduce_max(image),
                    tf.reduce_min(image)
                )
            )
            image = tf.multiply(image, tf.cast(self.threshold_value, image.dtype))
            input_features[key] = image

        return input_features

    def pre_process(self, input_features):
        _check_keys_(input_features, self.image_keys)
        for key in self.image_keys:
            image = input_features[key]
            min_value = np.min(image)
            max_value = np.max(image)
            image = (image - min_value) / (max_value - min_value)
            image = image * self.threshold_value
            input_features[key] = image
        return input_features

    def post_process(self, input_features):
        return input_features


class Random_Crop_and_Resize(ImageProcessor):
    def __init__(self, min_scale=0.80, image_rows=512, image_cols=512, image_keys=('image', 'annotation'),
                 interp=('bilinear', 'nearest'), dtypes=('float16', 'float16'), preserve_aspect_ratio=True):
        '''
        :param min_scale:
        :param image_rows:
        :param image_cols:
        :param image_keys:
        :param interp:
        :param preserve_aspect_ratio:
        '''
        self.image_rows = tf.constant(image_rows)
        self.image_cols = tf.constant(image_cols)
        self.scales = np.linspace(start=min_scale, stop=1.0, num=20)
        self.image_keys = image_keys
        self.interp = interp
        self.dtypes = dtypes
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        random_index = np.random.randint(0, len(self.scales))
        scale = self.scales[random_index]
        if scale != 1.0:
            for key, method, dtype in zip(self.image_keys, self.interp, self.dtypes):
                croppped_img = tf.image.central_crop(input_features[key], scale)
                croppped_img = tf.image.resize(croppped_img, size=(self.image_rows, self.image_cols),
                                               method=method, preserve_aspect_ratio=self.preserve_aspect_ratio)
                input_features[key] = tf.cast(croppped_img, dtype=dtype)

        return input_features


class Random_Left_Right_flip(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation')):
        self.image_keys = image_keys

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        do_aug = tf.random.uniform([]) > 0.5
        for key in self.image_keys:
            input_features[key] = tf.cond(do_aug, lambda: tf.image.flip_left_right(input_features[key]),
                                          lambda: input_features[key])

        return input_features


class Random_Contrast(ImageProcessor):
    def __init__(self, image_keys=('image',), lower_bounds=(0.90,), upper_bounds=(1.10,)):
        self.image_keys = image_keys
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for image_key, lower_bound, upper_bound in zip(self.image_keys, self.lower_bounds, self.upper_bounds):
            input_features[image_key] = tf.image.random_contrast(input_features[image_key],
                                                                 lower=lower_bound, upper=upper_bound)
        return input_features


class Random_Up_Down_flip(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation')):
        self.image_keys = image_keys

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        do_aug = tf.random.uniform([]) > 0.5
        for key in self.image_keys:
            input_features[key] = tf.cond(do_aug, lambda: tf.image.flip_up_down(input_features[key]),
                                          lambda: input_features[key])
        return input_features


class Extract_Patch(ImageProcessor):
    def __init__(self, image_key='image', annotation_key='annotation', box_key='bounding_box',
                 patch_size=(32, 192, 192)):
        self.image_key = image_key
        self.annotation_key = annotation_key
        self.box_key = box_key
        self.patch_size = patch_size

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, (self.image_key, self.annotation_key, self.box_key))

        image = input_features[self.image_key]
        annotation = input_features[self.annotation_key]
        # min_slice, max_slice, min_row, max_row, min_col, max_col
        bounding_box = input_features[self.box_key]

        # get random index inside bounding box of labels to have more regions without labels
        i_slice, i_row, i_col = tf.random.uniform(shape=[], minval=bounding_box[0], maxval=bounding_box[1] + 1,
                                                  dtype=tf.int64), \
                                tf.random.uniform(shape=[], minval=bounding_box[2], maxval=bounding_box[3] + 1,
                                                  dtype=tf.int64), \
                                tf.random.uniform(shape=[], minval=bounding_box[4], maxval=bounding_box[5] + 1,
                                                  dtype=tf.int64)

        zero = tf.constant(0, dtype=tf.int64)
        img_shape = tf.shape(image, out_type=tf.int64)
        z_start = tf.maximum(zero, i_slice - int(self.patch_size[0] / 2))
        z_stop = tf.minimum(i_slice + int(self.patch_size[0] / 2), img_shape[0])
        r_start = tf.maximum(zero, i_row - int(self.patch_size[1] / 2))
        r_stop = tf.minimum(i_row + int(self.patch_size[1] / 2), img_shape[1])
        c_start = tf.maximum(zero, i_col - int(self.patch_size[2] / 2))
        c_stop = tf.minimum(i_col + int(self.patch_size[2] / 2), img_shape[2])

        # pull patch size at the random index for both images and return the patch
        image = image[z_start:z_stop, r_start:r_stop, c_start:c_stop, ...]
        annotation = annotation[z_start:z_stop, r_start:r_stop, c_start:c_stop, ...]

        cropped_img_shape = tf.shape(image)
        remain_z, remain_r, remain_c = self.patch_size[0] - cropped_img_shape[0], \
                                       self.patch_size[1] - cropped_img_shape[1], \
                                       self.patch_size[2] - cropped_img_shape[2]

        paddings = [[tf.math.floor(remain_z / 2), tf.math.ceil(remain_z / 2)],
                    [tf.math.floor(remain_r / 2), tf.math.ceil(remain_r / 2)],
                    [tf.math.floor(remain_c / 2), tf.math.ceil(remain_c / 2)], [0, 0]]

        image = tf.pad(image, paddings=paddings, constant_values=tf.reduce_min(image))
        annotation = tf.pad(annotation, paddings=paddings, constant_values=tf.reduce_min(image))

        input_features[self.image_key] = image
        input_features[self.annotation_key] = annotation
        return input_features


class Random_Rotation(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation'), interp=('bilinear', 'nearest'), angle=0.25,
                 dtypes=('float16', 'float16'), filling=('nearest', 'constant')):
        '''
        :param image_keys:
        :param interp:
        :param angle:
        '''
        self.image_keys = image_keys
        self.interp = interp
        self.angle = angle
        self.dtypes = dtypes
        self.filling = filling

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        angles = tf.random.uniform(shape=[], minval=-self.angle, maxval=self.angle)
        for key, interp, dtype, filling in zip(self.image_keys, self.interp, self.dtypes, self.filling):
            images = tf.cast(input_features[key], dtype='float32')
            images = tfa.image.rotate(images=images, angles=angles,
                                      interpolation=interp, fill_mode=filling,
                                      fill_value=tf.math.reduce_min(images))
            input_features[key] = tf.cast(images, dtype=dtype)

        return input_features


class Keep_Connected_to_Mask(ImageProcessor):
    def __init__(self, prediction_keys=('prediction',), mask_keys=('og_annotation',), max_comp=2, min_vol=15000):
        '''
        This function keeps the prediction components connected to the mask. max_comp == 2 can be use whem mask
        represent both lungs for example
        '''
        self.prediction_keys = prediction_keys
        self.mask_keys = mask_keys
        self.max_comp = max_comp
        self.min_vol = min_vol

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.prediction_keys)
        for prediction_key, mask_key in zip(self.prediction_keys, self.mask_keys):
            global_pred = copy.deepcopy(input_features[prediction_key])
            global_pred = np.squeeze(global_pred)
            guided_pred = global_pred + input_features[mask_key]
            guided_pred[guided_pred > 0] = 1
            filtered_guide = np.zeros_like(guided_pred)
            for class_id in range(1, global_pred.shape[-1]):
                filtered_guide[..., class_id] = extract_main_component(nparray=guided_pred[..., class_id], dist=None,
                                                                       max_comp=self.max_comp, min_vol=self.min_vol)
            global_pred = global_pred * filtered_guide
            input_features[prediction_key] = global_pred
        return input_features


class ProcessPrediction(ImageProcessor):
    def __init__(self, threshold={}, connectivity={}, extract_main_comp={}, prediction_keys=('prediction',),
                 thread_count=int(cpu_count() / 2), dist={}, max_comp={}, min_vol={}):
        self.threshold = threshold
        self.connectivity = connectivity
        self.prediction_keys = prediction_keys
        self.extract_main_comp = extract_main_comp
        self.thread_count = thread_count
        self.dist = dist
        self.max_comp = max_comp
        self.min_vol = min_vol

    def worker_def(self, A):
        q = A
        while True:
            item = q.get()
            if item is None:
                break
            else:
                iteration, class_id = item
                # print('{}, '.format(iteration))
                try:
                    threshold_val = self.threshold.get(str(class_id))
                    connectivity_val = self.connectivity.get(str(class_id))
                    extract_main_comp_val = self.extract_main_comp.get(str(class_id))

                    if threshold_val != 0.0:
                        pred_id = self.global_pred[..., class_id]
                        pred_id[pred_id < threshold_val] = 0
                        pred_id[pred_id > 0] = 1
                        pred_id = pred_id.astype('int')

                        if extract_main_comp_val:
                            dist = self.dist.get(str(class_id))
                            max_comp = self.max_comp.get(str(class_id))
                            min_vol = self.min_vol.get(str(class_id))
                            pred_id = extract_main_component(nparray=pred_id, dist=dist, max_comp=max_comp,
                                                             min_vol=min_vol)

                        if connectivity_val:
                            main_component_filter = Remove_Smallest_Structures()
                            pred_id = main_component_filter.remove_smallest_component(pred_id)
                        self.global_pred[..., class_id] = pred_id
                except:
                    print('failed on class {}, '.format(iteration))
                q.task_done()

    def single_process(self):
        for class_id in range(1, self.global_pred.shape[-1]):  # ignore the first class as it is background
            threshold_val = self.threshold.get(str(class_id))
            connectivity_val = self.connectivity.get(str(class_id))
            extract_main_comp_val = self.extract_main_comp.get(str(class_id))

            if threshold_val != 0.0:
                pred_id = self.global_pred[..., class_id]
                pred_id[pred_id < threshold_val] = 0
                pred_id[pred_id > 0] = 1
                pred_id = pred_id.astype('int')

                if extract_main_comp_val:
                    dist = self.dist.get(str(class_id))
                    max_comp = self.max_comp.get(str(class_id))
                    min_vol = self.min_vol.get(str(class_id))
                    pred_id = extract_main_component(nparray=pred_id, dist=dist, max_comp=max_comp, min_vol=min_vol)

                if connectivity_val:
                    main_component_filter = Remove_Smallest_Structures()
                    pred_id = main_component_filter.remove_smallest_component(pred_id)
                self.global_pred[..., class_id] = pred_id

    def multi_process(self):
        # init threads
        q = Queue(maxsize=self.thread_count)
        threads = []
        for worker in range(self.thread_count):
            t = Thread(target=self.worker_def, args=(q,))
            t.start()
            threads.append(t)

        iteration = 1
        for class_id in range(1, self.global_pred.shape[-1]):  # ignore the first class as it is background
            item = [iteration, class_id]
            iteration += 1
            q.put(item)

        for i in range(self.thread_count):
            q.put(None)
        for t in threads:
            t.join()

    def pre_process(self, input_features):
        _check_keys_(input_features=input_features, keys=self.prediction_keys)
        for key in self.prediction_keys:
            self.global_pred = copy.deepcopy(input_features[key])
            self.global_pred = np.squeeze(self.global_pred)

            if self.thread_count > 1:
                self.multi_process()
            else:
                self.single_process()

            input_features[key] = self.global_pred
        return input_features


class Random_Translation(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation'), interp=('bilinear', 'nearest'),
                 translation_x=0.0, translation_y=0.0, dtypes=('float16', 'float16'), filling=('nearest', 'constant')):
        self.image_keys = image_keys
        self.interp = interp
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.dtypes = dtypes
        self.filling = filling

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        tx = tf.random.uniform(shape=[], minval=-self.translation_x, maxval=self.translation_x)
        ty = tf.random.uniform(shape=[], minval=-self.translation_y, maxval=self.translation_y)
        for key, interp, dtype, filling in zip(self.image_keys, self.interp, self.dtypes, self.filling):
            # for fill_value and tf.math.reduce_min needs float32 input
            images = tf.cast(input_features[key], dtype='float32')
            images = tfa.image.translate(images=images, translations=[tx, ty],
                                         interpolation=interp, fill_mode=filling,
                                         fill_value=tf.math.reduce_min(images))
            input_features[key] = tf.cast(images, dtype=dtype)

        return input_features


class Central_Crop_Img(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation')):
        self.image_keys = image_keys

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key in self.image_keys:
            input_features[key] = tf.image.central_crop(image=input_features[key], central_fraction=0.8)
        return input_features


class Threshold_Using_Median(ImageProcessor):
    def __init__(self, image_keys=('image',), dtypes=('float16',)):
        '''
        :param image_keys: image input of [row, col, 1]
        '''
        self.image_keys = image_keys
        self.dtypes = dtypes

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key, dtype in zip(self.image_keys, self.dtypes):
            image = tf.cast(input_features[key], dtype='float32')
            unique_values = tf.unique(tf.reshape(image, [-1]))
            # mid_index = unique_values.y.get_shape()[0] // 2
            # median_values = tf.reduce_min(tf.nn.top_k(unique_values.y, mid_index, sorted=False).values)
            median_values = tfp.stats.percentile(unique_values.y, q=50.)
            mask = tf.math.greater(image, median_values)
            binary = tf.cast(tf.where(mask, 1, 0), dtype=image.dtype)
            image = tf.math.multiply(image, binary)
            input_features[key] = tf.cast(image, dtype=dtype)

        return input_features


class Replace_Padding_Value(ImageProcessor):
    def __init__(self, image_keys=('image',), dtypes=('float16',), padding_value=0):
        '''
        :param image_keys: image input of [row, col, 1]
        '''
        self.image_keys = image_keys
        self.dtypes = dtypes
        self.padding_value = tf.constant([padding_value], dtype='float32')

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key, dtype in zip(self.image_keys, self.dtypes):
            image = tf.cast(input_features[key], dtype='float32')
            padding_mask = tf.math.equal(image, self.padding_value)
            unique_values = tf.unique(tf.reshape(image, [-1]))
            min_not_zero = tf.reduce_max(
                tf.negative(tf.nn.top_k(tf.negative(unique_values.y), k=2, sorted=False).values))
            image = tf.cast(tf.where(padding_mask, min_not_zero, image), dtype=image.dtype)
            input_features[key] = tf.cast(image, dtype=dtype)

        return input_features


class Normalize_Raw_Images(ImageProcessor):
    def __init__(self, image_key='image', mask_key='mask', force_expand=True):
        '''
        :param image_keys: image input of [row, col, 1]
        '''
        self.image_key = image_key
        self.mask_key = mask_key
        self.force_expand = force_expand

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, (self.image_key, self.mask_key,))

        image = tf.cast(input_features[self.image_key], dtype='float32')
        binary_image = tf.cast(input_features[self.mask_key], dtype='float32')

        if self.force_expand:
            binary_image = tf.expand_dims(binary_image, axis=-1)
        else:
            binary_image = tf.squeeze(binary_image, axis=0)

        cond_mask = tf.math.greater(binary_image, tf.constant([0], dtype=image.dtype))
        masked_image_values = tf.boolean_mask(image, cond_mask)

        # get mean std inside the masked image
        mean_inside = tf.math.reduce_mean(masked_image_values)
        std_inside = tf.math.reduce_std(masked_image_values)

        # apply std z
        stdz_image = tf.math.divide(
            tf.math.subtract(
                image,
                mean_inside
            ),
            std_inside
        )

        # mask std z image
        stdz_image = tf.multiply(stdz_image, binary_image)
        masked_image_values = tf.boolean_mask(stdz_image, cond_mask)

        # offset min
        min_inside = tf.math.reduce_min(masked_image_values)
        stdz_image = tf.math.add(
            stdz_image,
            tf.math.abs(min_inside)
        )

        stdz_image = tf.multiply(stdz_image, binary_image)
        masked_image_values = tf.boolean_mask(stdz_image, cond_mask)
        min_inside = tf.math.reduce_min(masked_image_values)
        max_inside = tf.math.reduce_max(masked_image_values)

        # apply min max norm
        image = tf.math.divide(
            tf.math.subtract(
                stdz_image,
                min_inside
            ),
            tf.math.subtract(
                max_inside,
                min_inside
            )
        )

        input_features[self.image_key] = image

        return input_features


class Hist_Equal(ImageProcessor):
    def __init__(self, image_keys=('image',), bins=255001):
        self.image_keys = image_keys
        self.bins = bins

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for image_key in self.image_keys:
            # input_features[image_key] = tfa.image.equalize(input_features[image_key])
            image = input_features[image_key]
            image = tf.cast(image, dtype='float32') # put that here cause float16 to int32 cause weird end values
            image = tf.math.multiply(image, tf.constant(1000.0, dtype=image.dtype))
            image = tf.cast(image, dtype='int32')

            # Compute the histogram of the image channel.
            histo = tf.histogram_fixed_width(image, [0, tf.math.reduce_max(image)], nbins=self.bins)

            # For the purposes of computing the step, filter out the nonzeros.
            nonzero_histo = tf.boolean_mask(histo, histo != 0)
            step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // (self.bins - 1)

            # If step is zero, return the original image.  Otherwise, build
            # lut from the full histogram and step and then index from it.
            if step == 0:
                result = image
            else:
                lut_values = (tf.cumsum(histo, exclusive=True) + (step // 2)) // step
                lut_values = tf.clip_by_value(lut_values, 0, (self.bins - 1))
                result = tf.gather(lut_values, image)

            image = tf.math.divide(tf.cast(result, dtype='float32'), tf.constant(1000.0, dtype='float32'))
            input_features[image_key] = image

        return input_features


class Binarize_And_Remove_Unconnected(ImageProcessor):
    def __init__(self, image_keys=('image',), dtypes=('float16',)):
        '''
        :param image_keys: image input of [row, col, 1]
        '''
        self.image_keys = image_keys
        self.dtypes = dtypes

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key, dtype in zip(self.image_keys, self.dtypes):
            image = tf.cast(input_features[key], dtype='float32')
            mask = tf.math.greater(image, tf.constant([0], dtype=image.dtype))
            binary = tf.where(mask, 1, 0)
            # label id are classified by first apparition
            binary = tf.expand_dims(binary, axis=0)
            filters = tf.ones((7, 7, binary.get_shape()[3]), dtype='int32')
            binary = tf.nn.dilation2d(binary, filters=filters, strides=(1, 1, 1, 1), dilations=(1, 1, 1, 1),
                                      padding="SAME", data_format="NHWC")
            binary = tf.cast(tf.math.divide(
                tf.math.subtract(
                    binary,
                    tf.reduce_min(binary)
                ),
                tf.math.subtract(
                    tf.reduce_max(binary),
                    tf.reduce_min(binary)
                )
            ), dtype='int32')
            binary = tf.squeeze(binary, axis=-1)
            # we remove the last axis because of the 3D (N, H, W) `Tensor` required input
            labels = tfa.image.connected_components(binary)
            # make sure we remove the backgound again
            labels = tf.math.multiply(labels, binary)
            volume_max = tf.cast(0, dtype='float32')
            keep_id = 0
            for label_id in range(tf.reduce_min(labels) + 1, tf.reduce_max(labels) + 1):
                volume_id = tf.reduce_sum(tf.cast(tf.math.equal(labels, [label_id]), tf.float32))
                if volume_id > volume_max:
                    keep_id = label_id
                    volume_max = volume_id

            # WARNING always specify the tf.squeeze axis otherwise tensor.shape.ndims may be None
            mask = tf.math.equal(tf.expand_dims(tf.squeeze(labels, axis=0), axis=-1), [keep_id])
            binary = tf.cast(tf.where(mask, 1, 0), dtype=image.dtype)
            image = tf.math.multiply(image, binary)

            input_features[key] = tf.cast(image, dtype=dtype)

        return input_features


class Distribute_into_3D_with_Mask(ImageProcessor):
    def __init__(self, min_z=0, max_z=np.inf, max_rows=np.inf, max_cols=np.inf, mirror_small_bits=True,
                 chop_ends=False, desired_val=1):
        self.max_z = max_z
        self.min_z = min_z
        self.max_rows, self.max_cols = max_rows, max_cols
        self.mirror_small_bits = mirror_small_bits
        self.chop_ends = chop_ends
        self.desired_val = desired_val

    def pre_process(self, input_features):
        out_features = OrderedDict()
        start_chop = 0
        image_base = input_features['image']
        annotation_base = input_features['annotation']
        mask_base = input_features['mask']
        image_path = input_features['image_path']
        z_images_base, rows, cols = image_base.shape
        if self.max_rows != np.inf:
            rows = min([rows, self.max_rows])
        if self.max_cols != np.inf:
            cols = min([cols, self.max_cols])
        image_base, annotation_base, mask_base = image_base[:, :rows, :cols], annotation_base[:, :rows,
                                                                              :cols], mask_base[:, :rows, :cols]
        step = min([self.max_z, z_images_base])
        for index in range(z_images_base // step + 1):
            image_features = OrderedDict()
            if start_chop >= z_images_base:
                continue
            image = image_base[start_chop:start_chop + step, ...]
            annotation = annotation_base[start_chop:start_chop + step, ...]
            mask = mask_base[start_chop:start_chop + step, ...]
            start_chop += step
            if image.shape[0] < max([step, self.min_z]):
                if self.mirror_small_bits:
                    while image.shape[0] < max([step, self.min_z]):
                        mirror_image = np.flip(image, axis=0)
                        mirror_annotation = np.flip(annotation, axis=0)
                        mirror_mask = np.flip(mask, axis=0)
                        image = np.concatenate([image, mirror_image], axis=0)
                        annotation = np.concatenate([annotation, mirror_annotation], axis=0)
                        mask = np.concatenate([mask, mirror_mask], axis=0)
                    image = image[:max([step, self.min_z])]
                    annotation = annotation[:max([step, self.min_z])]
                    mask = mask[:max([step, self.min_z])]
                elif self.chop_ends:
                    continue
            start, stop = _get_start_stop_(annotation, extension=0, desired_val=self.desired_val)
            if start == -1 or stop == -1:
                continue  # no annotation here
            image_features['image_path'] = image_path
            image_features['image'] = image
            image_features['annotation'] = annotation
            image_features['mask'] = mask
            image_features['start'] = start
            image_features['stop'] = stop
            for key in input_features.keys():
                if key not in image_features.keys():
                    image_features[key] = input_features[key]  # Pass along all other keys.. be careful
            out_features['Image_{}'.format(index)] = image_features
        input_features = out_features
        return input_features


def _get_start_stop_(annotation, extension=np.inf, desired_val=1):
    if len(annotation.shape) > 3:
        annotation = np.argmax(annotation, axis=-1)
    non_zero_values = np.where(np.max(annotation, axis=(1, 2)) >= desired_val)[0]
    start, stop = -1, -1
    if non_zero_values.any():
        start = int(non_zero_values[0])
        stop = int(non_zero_values[-1])
        start = max([start - extension, 0])
        stop = min([stop + extension, annotation.shape[0]])
    return start, stop
