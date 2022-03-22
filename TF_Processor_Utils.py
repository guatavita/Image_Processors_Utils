import sys, os
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from Image_Processor_Utils import *


def _check_keys_(input_features, keys):
    if type(keys) is list or type(keys) is tuple:
        for key in keys:
            assert key in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                                 '{} was not found'.format(key)
    else:
        assert keys in input_features.keys(), 'Make sure the key you are referring to is present in the features, ' \
                                              '{} was not found'.format(keys)


class RandomNoise(ImageProcessor):
    def __init__(self, input_keys=('image',), max_noise=2.5):
        '''
        Return the image feature with an additive noise randomly weighted between [0.0, max_noise)
        :param max_noise: maximum magnitude of the noise in HU (apply before normalization)
        '''
        self.input_keys = input_keys
        self.max_noise = max_noise

    def parse(self, image_features, *args, **kwargs):
        _check_keys_(input_features=image_features, keys=self.input_keys)
        for key in self.input_keys:
            data = image_features[key]
            dtype = data.dtype
            data = tf.cast(data, 'float32')
            data += tf.random.uniform(shape=[], minval=0.0, maxval=self.max_noise,
                                      dtype='float32') * tf.random.normal(tf.shape(data),
                                                                          mean=0.0, stddev=1.0, dtype='float32')
            data = tf.cast(data, dtype)
            image_features[key] = data
        return image_features


class DeepCopyKey(ImageProcessor):
    def __init__(self, from_keys=('annotation',), to_keys=('annotation_original',)):
        self.from_keys, self.to_keys = from_keys, to_keys

    def parse(self, image_features):
        _check_keys_(input_features=image_features, keys=self.from_keys)
        for from_key, to_key in zip(self.from_keys, self.to_keys):
            image_features[to_key] = tf.identity(image_features[from_key])
        return image_features


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
        annotation = tf.pad(annotation, paddings=paddings, constant_values=tf.reduce_min(annotation))

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
    def __init__(self, image_key='image', mask_key='mask', force_expand=True, force_squeeze=False):
        '''
        :param image_keys: image input of [row, col, 1]
        '''
        self.image_key = image_key
        self.mask_key = mask_key
        self.force_expand = force_expand
        self.force_squeeze = force_squeeze

    def parse(self, input_features, *args, **kwargs):
        _check_keys_(input_features, (self.image_key, self.mask_key,))

        image = tf.cast(input_features[self.image_key], dtype='float32')
        binary_image = tf.cast(input_features[self.mask_key], dtype='float32')

        if self.force_expand:
            binary_image = tf.expand_dims(binary_image, axis=-1)

        if self.force_squeeze:
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
            image = tf.cast(image, dtype='float32')  # put that here cause float16 to int32 cause weird end values
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
