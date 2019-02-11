
import pandas as pd
import numpy as np
import logging
import threading
import os

import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TARGET_SIZE = (224, 224, 3)


def _preprocess_input(_img):
    # [-1,1] range
    _img /= 127.5
    _img -= 1.
    return _img


class TripletGenerator(Sequence):
    def __init__(self,
                 file_path,
                 image_path,
                 image_test_path,
                 nb_classes_batch,
                 nb_images_per_class_batch,
                 target_size=TARGET_SIZE,
                 subset='training',
                 validation_split=0.0,
                 seed=42,
                 is_reptile=False,
                 shuffle=True,
                 excluded_classes=['new_whale'],
                 class_weight_type=None,
                 **kwargs):
        self.file_path = file_path
        self.shuffle = shuffle
        self.target_size = target_size
        self.nb_classes_batch = nb_classes_batch
        self.nb_images_per_class_batch = nb_images_per_class_batch
        self.batch_size = nb_classes_batch * nb_images_per_class_batch
        self.image_path = image_path
        self.image_test_path = image_test_path
        self.df = pd.read_csv(file_path)
        self.index_array = None
        self.lock = threading.Lock()
        self.is_reptile = is_reptile

        logger.info('exclude_class: {}'.format(excluded_classes))
        logger.info('class_weight_type: {}'.format(class_weight_type))

        blacklisted_class_ix = self.df['Id'].isin(excluded_classes)
        logger.info("{} instances excluded".format(np.sum(blacklisted_class_ix)))
        df = self.df[~blacklisted_class_ix]

        classes = list(df['Id'].unique())
        self.class_indices = dict(zip(classes, range(len(classes))))
        self.class_inv_indices = dict(zip(range(len(classes)), classes))

        train_classes, test_classes = train_test_split(classes, test_size=validation_split, random_state=seed)
        if subset == 'training':
            self.df = df[df['Id'].isin(train_classes)]
        else:
            self.df = df[df['Id'].isin(test_classes)]
        logger.info("data has shape: {}".format(self.df.shape))

        classes = self.df['Id'].values
        self.classes = np.array([self.class_indices[cls] for cls in classes])
        self.filenames = np.array(self.df['Image'])

        self.class_indices_subset = {k: v for k, v in self.class_inv_indices.items() if k in self.classes}

        kwargs.update({'preprocessing_function': _preprocess_input})
        logger.info(kwargs)
        self.image_generator = ImageDataGenerator(**kwargs)
        # test time generator
        self.image_generator_inference = ImageDataGenerator(preprocessing_function=_preprocess_input)

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

    def __len__(self):
        """
        number of steps per epoch
        number of classes in subet / nb_classes_batch
        :return: 
        """
        return len(self.class_indices_subset) // self.nb_classes_batch

    def get_nb_classes(self):
        """
        number of all classes
        :return: 
        """
        return len(self.class_indices)

    def _set_index_array(self):

        self.index_array = list(self.class_indices_subset.keys())
        if self.shuffle:
            self.index_array = np.random.permutation(self.index_array)

    def on_epoch_end(self):
        """
        shuffle at epoch end
        """
        self._set_index_array()

    def get_train_image_from_class_ix(self, ixs):
        return self._get_batches_of_transformed_samples(ixs)

    def __getitem__(self, idx):
        if self.index_array is None:
            self._set_index_array()
        # unique classes to pull
        index_array = self.index_array[idx * self.nb_classes_batch: self.nb_classes_batch * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """
        validation and training behaves the same way w.r.t. augmentation
        :param index_array: 
        :return: 
        """
        batch_x = []
        for i, class_ix in enumerate(index_array):
            _samples = np.zeros(tuple([self.nb_images_per_class_batch] + list(self.target_size)), dtype='float32')
            filenames = self.filenames[self.classes == class_ix]
            if len(filenames) > self.nb_images_per_class_batch:
                np.random.shuffle(filenames)
                filenames = filenames[:self.nb_images_per_class_batch]
            logger.debug("{} files for class {}".format(len(filenames), class_ix))
            for j, filename in enumerate(filenames):
                img = load_img(os.path.join(self.image_path, filename),
                               target_size=self.target_size)
                x = img_to_array(img, data_format='channels_last')
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(img, 'close'):
                    img.close()
                _samples[j] = self.image_generator.standardize(x)

            # require at least `nb_images_per_class_batch` per class
            nb_missing = _samples.shape[0] - j - 1
            # select images to transform. No need to standardize again.
            img_ix_transformed = np.random.choice(j + 1, nb_missing)
            for img_ix, k in zip(img_ix_transformed, range(nb_missing)):
                x = _samples[img_ix]
                params = self.image_generator.get_random_transform(self.target_size)
                x = self.image_generator.apply_transform(x, params)
                _samples[j + k + 1] = x
            logger.debug("{} transformations for class {}".format(nb_missing, class_ix))
            batch_x.append(_samples)
        batch_x = np.vstack(batch_x)
        # build batch of labels
        _labels = range(10) if self.is_reptile else index_array
        batch_y = np.repeat(_labels, self.nb_images_per_class_batch)
        if self.is_reptile:
            batch_y = to_categorical(batch_y, num_classes=10)
        assert len(batch_y), len(batch_x)
        return batch_x, batch_y

    def get_test_generator(self, x):
        """
        :param x: test data
        :return: NumpyArrayIterator
        """
        return self.image_generator_inference.flow(x=x,
                                                   shuffle=False,
                                                   batch_size=self.batch_size)






