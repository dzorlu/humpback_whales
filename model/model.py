

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import top_k_categorical_accuracy


def top_5_accuracy(x, y): return top_k_categorical_accuracy(x, y, 5)


def create_model_fn(params):
    """
    create model function and callbacks given the params
    :return:
    """
    if params.image_dim[0] not in [224, 128]:
        ValueError('hip to be square..')
    if params.model_architecture == 'mobilenet':
        base_model = tf.keras.applications.mobilenet.MobileNet(input_shape=params.image_dim,
                                                               alpha=1.0,
                                                               depth_multiplier=1,
                                                               dropout=params.dropout,
                                                               include_top=True,
                                                               weights=None,
                                                               classes=params.nb_classes,
                                                               input_tensor=None,
                                                               pooling=None)
    elif params.model_architecture == 'resnet':
        base_model = tf.keras.applications.resnet50.ResNet50(input_shape=params.image_dim,
                                                             include_top=True,
                                                             weights=None,
                                                             classes=params.nb_classes)
    else:
        raise ValueError("architecture not defined.")
    base_model.compile(optimizer=Adam(lr=params.lr_rate),
                       loss='categorical_crossentropy',
                       metrics=['categorical_accuracy', top_5_accuracy])
    tf.logging.info(base_model.summary())
    return base_model