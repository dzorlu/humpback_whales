import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import pandas as pd

import os
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224, 3)


# lot of classes with a single image
# grey-scale images (50%)
def is_grayscale(_img):
    return np.diff(_img, n=2, axis=-1).sum() > 1


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def _preprocess_input(_img):
    # [-1,1] range
    _img /= 127.5
    _img -= 1.
    return _img

def train_preprocess_fn(_img):
    # turn into grayscale
    if not is_grayscale(_img) and np.random.uniform(0, 1) > 0.5:
        _img = rgb2gray(_img)
        _img = np.repeat(_img, 3).reshape(_img.shape + (3,))
    _preprocess_input(_img)


class Generator(object):
    def __init__(self,
                 file_path,
                 image_path,
                 image_test_path,
                 batch_size,
                 target_size=TARGET_SIZE,
                 validation_split=0.0,
                 excluded_classes=['new_whale'],
                 class_weight_type=None,
                 **kwargs):
        self.target_size = target_size
        self._validation_split = validation_split
        self.batch_size = batch_size
        self.image_path = image_path
        self.image_test_path = image_test_path
        self.df = pd.read_csv(file_path)


        #self.X_test, self.image_names_test = _get_test_images_and_names(image_test_path, self.target_size)

        logger.info('exclude_class: {}'.format(excluded_classes))
        logger.info('class_weight_type: {}'.format(class_weight_type))

        blacklisted_class_ix = self.df['Id'].isin(excluded_classes)
        self.df = self.df[~blacklisted_class_ix]
        logger.info("data has shape: {}".format(self.df.shape))

        classes = list(self.df['Id'].unique())
        # class_name -> int
        self.class_indices = dict(zip(classes, range(len(classes))))
        # int -> class_name
        self.class_inv_indices = dict(zip(range(len(classes)), classes))
        classes = self.df['Id'].values

        self.classes = np.array([self.class_indices[cls] for cls in classes])
        self._class_weights = self.calculate_class_weights(class_weight_type)

        # pass preprocess_fn and validation. Keras fn applies transformations to val data too. ugh.
        kwargs.update({'preprocessing_function': _preprocess_input, 'validation_split': validation_split})
        logger.info(kwargs)
        self.image_generator = ImageDataGenerator(**kwargs)
        self.image_generator_inference = ImageDataGenerator(preprocessing_function=_preprocess_input)

    @property
    def validation_split(self):
        return self._validation_split

    def get_test_images_and_names(self):
        img_names = os.listdir(self.image_test_path)
        filepaths = [os.path.join(self.image_test_path, img_name) for img_name in img_names]
        _x = np.zeros((len(filepaths),) + self.target_size, dtype='float32')
        for i, _path in enumerate(filepaths):
            _img = load_img(_path, target_size=self.target_size)
            _x[i] = img_to_array(_img)
            if hasattr(_img, 'close'):
                _img.close()
        return _x, img_names

    def calculate_class_weights(self, _type=None):
        _counter = Counter(self.classes)
        if _type == 'log':
            w = np.array(list(_counter.values()))
            w = np.log(1 + w)
            w = max(w) / w
        elif _type == 'balanced':
            w = np.array(list(_counter.values()))
            w = max(w) / w
        else:
            return None
        return {k: v for k, v in zip(_counter.keys(), w)}

    def get_class_weights(self):
        return self._class_weights

    def get_nb_classes(self):
        return len(self.class_indices)

    def get_train_generator(self, shuffle=True):
        """ 
        :return: DataFrameIterator
        """
        return self.image_generator.flow_from_dataframe(dataframe=self.df,
                                                        directory=self.image_path,
                                                        x_col='Image',
                                                        y_col='Id',
                                                        target_size=self.target_size[:2],
                                                        class_mode='categorical',
                                                        subset='training',
                                                        shuffle=shuffle,
                                                        batch_size=self.batch_size)

    def get_eval_generator(self):
        """ 
        does augmentation too.
        :return: DataFrameIterator
        """
        return self.image_generator.flow_from_dataframe(dataframe=self.df,
                                                        directory=self.image_path,
                                                        x_col='Image',
                                                        y_col='Id',
                                                        target_size=self.target_size[:2],
                                                        class_mode='categorical',
                                                        subset='validation',
                                                        shuffle=True,
                                                        batch_size=self.batch_size)

    def get_test_generator(self, x):
        """
        :param x: test data
        :return: NumpyArrayIterator
        """
        return self.image_generator_inference.flow(x=x,
                                                   shuffle=False,
                                                   batch_size=self.batch_size)
