import os, copy
from _collections import OrderedDict

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tensorflow_addons.image import transform_ops

import numpy as np
import SimpleITK as sitk
from skimage import morphology, measure
from scipy.spatial import distance
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
import cv2
import math

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


def extract_main_component(nparray, dist=50, max_comp=2):
    # TODO create a dictionnary of the volume per label to filter the keep_id with max_comp

    labels = morphology.label(nparray, connectivity=3)
    temp_img = np.zeros(labels.shape)

    if np.max(labels) > 1:

        volumes = [labels[labels == i].shape[0] for i in range(1, labels.max() + 1)]
        max_val = volumes.index(max(volumes)) + 1

        keep_values = []
        keep_values.append(max_val)

        ref_volume = np.copy(labels)
        ref_volume[ref_volume != max_val] = 0
        ref_volume[ref_volume > 0] = 1
        ref_points = measure.marching_cubes(ref_volume, step_size=3, method='lewiner')[0]

        for i in range(1, labels.max() + 1):
            if i == max_val:
                continue

            # compute distance
            temp_volume = np.copy(labels)
            temp_volume[temp_volume != i] = 0
            temp_volume[temp_volume > 0] = 1

            try:
                # this remove small 'artifacts' cause they cannot be meshed
                temp_points = measure.marching_cubes(temp_volume, step_size=3, method='lewiner')[0]
            except:
                continue

            for ref_point in ref_points:
                distances = [distance.euclidean(ref_point, temp_point) for temp_point in temp_points]
                if min(distances) <= dist:
                    keep_values.append(i)
                    break

        # this is faster than 'temp_img[np.isin(labels, keep_values)] = 1' for most cases
        for values in keep_values:
            temp_img[labels == values] = 1
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

        label_image = self.Connected_Component_Filter.Execute(
            sitk.BinaryThreshold(sitk.Cast(annotation_handle, sitk.sitkFloat32), lowerThreshold=0.01,
                                 upperThreshold=np.inf))
        label_image = self.RelabelComponent.Execute(label_image)
        output = sitk.BinaryThreshold(sitk.Cast(label_image, sitk.sitkFloat32), lowerThreshold=0.1, upperThreshold=1.0)

        if convert:
            output = sitk.GetArrayFromImage(output)

        return output


class Focus_on_CT(ImageProcessor):
    def __init__(self, threshold_value=-250.0, mask_value=1, debug=False):
        # TODO this class needs to be cleaned
        self.threshold_value = threshold_value
        self.mask_value = mask_value
        self.bb_parameters = []
        self.final_padding = []
        self.original_shape = {}
        self.squeeze_flag = False
        self.debug = debug

    def pre_process(self, input_features):
        images = input_features['image']
        annotations = None
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

        if annotations != None:
            if annotations.dtype != 'float32':
                annotations = annotations.astype('float32')
            if self.squeeze_flag:
                annotations = np.squeeze(annotations)
            rescaled_input_label, self.final_padding = self.crop_resize_pad(input=annotations,
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
            min_slice, max_slice, min_row, max_row, min_col, max_col = compute_bounding_box(prediction[..., class_id],
                                                                                            padding=0)
            spacing = input_features['spacing']
            nb_slices = math.ceil(sup_margin / spacing[-1])
            new_prediction = np.zeros(prediction.shape[0:-1] + (prediction.shape[-1] + 1,), dtype=prediction.dtype)
            new_prediction[..., 0:prediction.shape[-1]] = prediction
            new_prediction[..., -1][(max_slice + 1 - nb_slices):(max_slice + 1), ...] = new_prediction[..., class_id][
                                                                                        (max_slice + 1 - nb_slices):(
                                                                                                    max_slice + 1), ...]
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
    def __init__(self, annotation_input=[1, 2], to_annotation=1):
        self.annotation_input = annotation_input
        self.to_annotation = to_annotation

    def pre_process(self, input_features):
        annotation = copy.deepcopy(input_features['annotation'])
        assert len(annotation.shape) == 3 or len(
            annotation.shape) == 4, 'To combine annotations the size has to be 3 or 4'
        if len(annotation.shape) == 3:
            for val in self.annotation_input:
                annotation[annotation == val] = self.to_annotation
        elif len(annotation.shape) == 4:
            annotation[..., self.to_annotation] += annotation[..., self.annotation_input]
            del annotation[..., self.annotation_input]
        input_features['mask'] = annotation
        return input_features


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


class Expand_Dimensions_Per_Key(ImageProcessor):
    def __init__(self, axis=-1, image_keys=('image',)):
        self.axis = axis
        self.image_keys = image_keys

    def parse(self, input_features, *args, **kwargs):
        for key in self.image_keys:
            input_features[key] = tf.expand_dims(input_features[key], axis=self.axis)
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


class Random_Crop_and_Resize(ImageProcessor):
    def __init__(self, min_scale=0.80, image_rows=512, image_cols=512, image_keys=('image', 'annotation'),
                 interp=('bilinear', 'nearest'), preserve_aspect_ratio=True):
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
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        random_index = np.random.randint(0, len(self.scales))
        scale = self.scales[random_index]
        if scale != 1.0:
            for key, method in zip(self.image_keys, self.interp):
                croppped_img = tf.image.central_crop(input_features[key], scale)
                input_features[key] = tf.image.resize(croppped_img, size=(self.image_rows, self.image_cols),
                                                      method=method, preserve_aspect_ratio=self.preserve_aspect_ratio)

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
    def __init__(self, image_key='image', annotation_key='annotation', patch_size=(32, 192, 192),
                 is_validation=False):
        self.image_key = image_key
        self.annotation_key = annotation_key
        self.patch_size = patch_size
        self.is_validation = is_validation

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, (self.image_key, self.annotation_key))

        image = input_features[self.image_key]
        annotation = input_features[self.annotation_key]

        if self.is_validation:
            # use np.where to have more overlap with labels
            slice_list, row_list, col_list = np.where(annotation > 0)
            slice_rind, row_rind, col_rind = np.random.randint(0, len(slice_list)), \
                                             np.random.randint(0, len(row_list)), \
                                             np.random.randint(0, len(col_list))
            i_slice, i_row, i_col = slice_list[slice_rind], row_list[row_rind], col_list[col_rind]
        else:
            # get random index inside bounding box of labels to have more regions without labels
            min_slice, max_slice, min_row, max_row, min_col, max_col = compute_bounding_box(annotation, padding=0)
            i_slice, i_row, i_col = np.random.randint(min_slice, max_slice + 1), \
                                    np.random.randint(min_row, max_row + 1), \
                                    np.random.randint(min_col, max_col + 1)

        # pull patch size at the random index for both images and return the patch
        image = image[i_slice - self.patch_size[0] / 2:i_slice + self.patch_size[0] / 2,
                i_row - self.patch_size[1] / 2:i_row + self.patch_size[1] / 2,
                i_col - self.patch_size[2] / 2:i_col + self.patch_size[2] / 2, ...]
        annotation = annotation[i_slice - self.patch_size[0] / 2:i_slice + self.patch_size[0] / 2,
                     i_row - self.patch_size[1] / 2:i_row + self.patch_size[1] / 2,
                     i_col - self.patch_size[2] / 2:i_col + self.patch_size[2] / 2, ...]

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


class ProcessPrediction(ImageProcessor):
    def __init__(self, threshold={}, connectivity={}, extract_main_comp={}, prediction_keys=('prediction',),
                 thread_count=int(cpu_count() / 2), dist=50, max_comp=2):
        self.threshold = threshold
        self.connectivity = connectivity
        self.prediction_keys = prediction_keys
        self.extract_main_comp = extract_main_comp
        self.thread_count = thread_count
        self.dist = dist
        self.max_comp = max_comp

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
                            pred_id = extract_main_component(nparray=pred_id, dist=self.dist, max_comp=self.max_comp)

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
                    pred_id = extract_main_component(nparray=pred_id, dist=self.dist, max_comp=self.max_comp)

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

            # WARNING always specify the tf.sqeeze axis otherwise tensor.shape.ndims may be None
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
