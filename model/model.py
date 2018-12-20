

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.contrib.losses.python.metric_learning import triplet_semihard_loss

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def top_5_accuracy(x, y): return top_k_categorical_accuracy(x, y, 5)


def create_model_fn(params):
    """
    create model function and callbacks given the params
    :return:
    """
    if params.image_dim[0] not in [224, 128]:
        ValueError('hip to be square..')
    if (params.nb_layers_to_freeze and not params.pretrained) or (params.nb_layers_to_freeze == 0 and params.pretrained):
        ValueError('set the pretrained to TRUE if nb_layers_to_freeze is specified')
    if params.loss == 'triplet_semihard_loss' and not params.embedding_hidden_dim:
        ValueError('set the embedding_hidden_dim if triplet_semihard_loss is specified')
    _include_top = True
    _weights = None
    if params.pretrained:
        logger.info('pretrained..')
        _include_top = False
        _weights = 'imagenet'
    if params.model_architecture == 'mobilenet':
        base_model = tf.keras.applications.mobilenet.MobileNet(input_shape=params.image_dim,
                                                               alpha=1.0,
                                                               depth_multiplier=1,
                                                               dropout=params.dropout,
                                                               include_top=_include_top,
                                                               weights=_weights,
                                                               classes=params.nb_classes,
                                                               input_tensor=None,
                                                               pooling=None)
        reshape_size = 1024
    elif params.model_architecture == 'resnet':
        base_model = tf.keras.applications.resnet50.ResNet50(input_shape=params.image_dim,
                                                             include_top=_include_top,
                                                             weights=_weights,
                                                             classes=params.nb_classes)
        reshape_size = 2048
    else:
        raise ValueError("architecture not defined.")
    if params.loss == 'triplet_semihard_loss':
        _loss = triplet_semihard_loss
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        shape = (1, 1, int(reshape_size * 1.0))
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(params.dropout, name='dropout')(x)
        # retrieve embedding
        x = layers.Conv2D(params.embedding_hidden_dim, (1, 1),
                          padding='same',
                          name='conv_embedding')(x)
        # l2 normalize
        x = layers.Reshape((params.embedding_hidden_dim,), name='reshape_2')(x)
        x = layers.Lambda(lambda _x: tf.keras.backend.l2_normalize(_x, axis=1))(x)
        # TODO: TENSORBOARD
        model = Model(inputs=base_model.input, outputs=x)
    elif params.pretrained:
        logger.info("append the top to the structure..")
        # TODO: COULD BE PRETRAINED AND TTIPLET
        _loss = 'categorical_crossentropy'
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        shape = (1, 1, int(reshape_size * 1.0))
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(params.dropout, name='dropout')(x)
        x = layers.Conv2D(params.nb_classes, (1, 1),
                          padding='same',
                          name='conv_preds')(x)
        x = layers.Reshape((params.nb_classes,), name='reshape_2')(x)
        x = layers.Activation('softmax', name='act_softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)
    else:
        _loss = 'categorical_crossentropy'
        model = base_model

    if params.nb_layers_to_freeze:
        for i, layer in enumerate(model.layers):
            if i < params.nb_layers_to_freeze:
                layer.trainable = False
        logger.info("{} out of {} layers frozen..".format(params.nb_layers_to_freeze, i))

    model.compile(optimizer=Adam(lr=params.lr_rate),
                  loss=_loss,
                  metrics=['categorical_accuracy', top_5_accuracy])
    tf.logging.info(model.summary())
    return model