import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

TARGET_SIZE = (224, 224, 3)


class Generator(ImageDataGenerator):
    def __init__(self,
                 file_path,
                 image_path,
                 is_train=True,
                 **kwargs):
        self.X = None
        self.y = None
        _x = []
        df = pd.read_csv(file_path)
        for img_name in df.image.values:
            _path = os.path.join(image_path, img_name)
            _img = load_img(_path, TARGET_SIZE)
            _x.append(_img)
        self.X = np.stack(_x)
        tf.logging.info("Generated %s Examples", len(self.X))
        if is_train:
            encoder = LabelEncoder()
            _y = df.id.values
            self.y = encoder.fit_transform(_y)
            tf.logging.info("Generated %s Labels", len(self.y))
        super(Generator, self).__init__(kwargs)
