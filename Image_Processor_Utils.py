import os
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tensorflow_addons.image import transform_ops

from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors import *


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


class Expand_Dimensions_Per_Key(ImageProcessor):
    def __init__(self, axis=-1, image_keys=('image', 'annotation'),):
        self.axis = axis
        self.image_keys = image_keys

    def pre_process(self, input_features, *args, **kwargs):
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

    def pre_process(self, input_features, *args, **kwargs):
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

    def pre_process(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key, dtype in zip(self.image_keys, self.dtypes):
            image = tf.cast(input_features[key], dtype='float32')
            image = tf.image.per_image_standardization(image=image)
            input_features[key] = tf.cast(image, dtype=dtype)

        return input_features


class Per_Image_MinMax_Normalization(ImageProcessor):
    def __init__(self, image_keys=('image',), threshold_value=255.0):
        '''
        :param image_keys:
        :param treshold_value:
        '''
        self.image_keys = image_keys
        self.threshold_value = threshold_value

    def pre_process(self, input_features, *args, **kwargs):
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

    def pre_process(self, input_features, *args, **kwargs):
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

    def pre_process(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key in self.image_keys:
            img_shape = tf.shape(input_features[key])
            if len(img_shape) == 4:
                random_var = tf.random.uniform([1], minval=0, maxval=2, dtype=tf.dtypes.int32)
                if random_var == 1:
                    input_features[key] = tf.image.flip_left_right(input_features[key])
            else:
                input_features[key] = tf.image.random_flip_left_right(input_features[key])

        return input_features

class Random_Up_Down_flip(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation')):
        self.image_keys = image_keys

    def pre_process(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key in self.image_keys:
            img_shape = tf.shape(input_features[key])
            if len(img_shape) == 4:
                random_var = tf.random.uniform([1], minval=0, maxval=2, dtype=tf.dtypes.int32)
                if random_var == 1:
                    input_features[key] = tf.image.flip_up_down(input_features[key])
            else:
                input_features[key] = tf.image.random_flip_up_down(input_features[key])

        return input_features


class Random_Rotation(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation'), interp=('bilinear', 'nearest'), angle=0.25):
        '''
        :param image_keys:
        :param interp:
        :param angle:
        '''
        self.image_keys = image_keys
        self.interp = interp
        self.angle = angle

    def pre_process(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        angles = tf.random.uniform(shape=[], minval=-self.angle, maxval=self.angle)
        for key, method in zip(self.image_keys, self.interp):
            input_features[key] = tfa.image.rotate(images=input_features[key], angles=angles, interpolation=method)

        return input_features


class Random_Translation(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation'), interp=('bilinear', 'nearest'),
                 translation_x=0.0, translation_y=0.0, dtypes=('float16', 'float16')):
        self.image_keys = image_keys
        self.interp = interp
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.dtypes = dtypes

    def pre_process(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        tx = tf.random.uniform(shape=[], minval=-self.translation_x, maxval=self.translation_x)
        ty = tf.random.uniform(shape=[], minval=-self.translation_y, maxval=self.translation_y)
        for key, method, dtype in zip(self.image_keys, self.interp, self.dtypes):
            # for some reasons this function needs float32 input
            images = tf.cast(input_features[key], dtype='float32')
            images = tfa.image.translate(images=images, translations=[tx, ty],
                                         interpolation=method, fill_mode='constant',
                                         fill_value=tf.math.reduce_min(images))
            input_features[key] = tf.cast(images, dtype=dtype)

        return input_features


class Central_Crop_Img(ImageProcessor):
    def __init__(self, image_keys=('image', 'annotation')):
        self.image_keys = image_keys

    def pre_process(self, input_features, *args, **kwargs):
        _check_keys_(input_features, self.image_keys)
        for key in self.image_keys:
            input_features[key] = tf.image.central_crop(image=input_features[key], central_fraction=0.5)
        return input_features


class Binarize_And_Remove_Unconnected(ImageProcessor):
    def __init__(self, image_keys=('image',), dtypes=('float16',)):
        '''
        :param image_keys: image input of [row, col, 1]
        '''
        self.image_keys = image_keys
        self.dtypes = dtypes

    def pre_process(self, input_features, *args, **kwargs):
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
