import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import logging
from collections import Counter
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224, 3)


# lot of classes with a single image
# grey-scale images (50%)
def is_grayscale(_img):
    return np.diff(_img, n=2, axis=-1).sum() > 1


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def preprocess_fn(_img):
    # turn into grayscale
    if not is_grayscale(_img) and np.random.uniform(0, 1) > 0.5:
        _img = rgb2gray(_img)
        return np.repeat(_img, 3).reshape(_img.shape + (3,))
    else:
        return _img


def calculate_class_weights(labels, _type=None):
    _counter = Counter(labels)
    if _type == 'log':
        chn = np.array(list(_counter.values()))
        chn_ = np.log(chn + 1) / np.sum(np.log(chn + 1))
        return {k: v for k, v in zip(_counter.keys(), chn_)}
    elif _type == 'uniform':
        _uniform = 1./len(_counter.values())
        return {k: _uniform for k, v in _counter.keys()}
    else:
        return None


def get_test_images_and_names(model_params):
    _x = []
    img_names = os.listdir(model_params.test_image_path)
    filepaths = [os.path.join(model_params.test_image_path, img_name) for img_name in img_names]
    for _path in filepaths:
        _img = load_img(_path, target_size=model_params.image_dim)
        _x.append(img_to_array(_img))
    return np.stack(_x), img_names


class Generator(object):
    def __init__(self,
                 file_path,
                 image_path,
                 batch_size,
                 validation_split=0.1,
                 random_state=42,
                 excluded_classes=['new_whale'],
                 class_weight_type='log',
                 **kwargs):
        logger.info('exclude_class: {}'.format(excluded_classes))
        logger.info('class_weight_type: {}'.format(class_weight_type))
        self.X = None
        self.y = None
        self.X_val = None
        self.y_val = None
        self.batch_size = batch_size
        _x, _y = [], []
        df = pd.read_csv(file_path)
        for _, _row in df.iterrows():
            if not excluded_classes or _row.Id not in excluded_classes:
                _path = os.path.join(image_path, _row.Image)
                _img = load_img(_path, target_size=TARGET_SIZE)
                _x.append(img_to_array(_img))
                _y.append(_row.Id)
        self.X = np.stack(_x)
        del _x

        encoder = LabelEncoder()
        _y = encoder.fit_transform(_y)
        self.encoder = encoder
        self.y = np.stack(_y)
        self._class_weights = calculate_class_weights(self.y, class_weight_type)

        print("Generated %s Examples", len(self.X))
        print("Generated %s Labels", len(self.y))
        if validation_split:
            self.X, self.X_val, self.y, self.y_val = train_test_split(self.X,
                                                                      self.y,
                                                                      test_size=validation_split,
                                                                      random_state=random_state)


        # pass preprocess_fn
        kwargs.update({'preprocessing_function': preprocess_fn})
        logger.info(kwargs)
        self.image_generator = ImageDataGenerator(**kwargs)
        self.image_generator.fit(self.X)
        # test data gen contains only normalization
        inference_params = {'featurewise_center': True,
                            'featurewise_std_normalization': True}
        self.image_generator_inference = ImageDataGenerator(**inference_params)
        self.image_generator_inference.fit(self.X)

    def get_class_weights(self):
        return self._class_weights

    def get_nb_classes(self):
        return len(self.encoder.classes_)

    def get_image_dim(self):
        return TARGET_SIZE

    def get_train_generator(self):
        """ 
        :return: NumpyArrayIterator
        """
        return self.image_generator.flow(x=self.X,
                                         y=self.y,
                                         shuffle=True,
                                         batch_size=self.batch_size)

    def get_eval_generator(self):
        """ 
        :return: NumpyArrayIterator
        """
        return self.image_generator.flow(x=self.X_val,
                                         y=self.y_val,
                                         shuffle=False,
                                         batch_size=self.batch_size)

    def get_test_generator(self, x):
        """
        :param x: test data
        :return: NumpyArrayIterator
        """
        return self.image_generator_inference.flow(x=x,
                                                   shuffle=False,
                                                   batch_size=self.batch_size)