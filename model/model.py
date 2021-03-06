

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


def create_model_fn(params, is_optimizer_adam=True):
    """
    create model function and callbacks given the params
    :return:
    """
    if params.image_dim[0] not in [224, 128] and params.pretrained:
        ValueError('hip to be square..')# need this for pretrained models.
    if (params.nb_layers_to_freeze and not params.pretrained) or (params.nb_layers_to_freeze == 0 and params.pretrained):
        ValueError('set the pretrained to TRUE if nb_layers_to_freeze is specified')
    if params.loss == 'triplet_semihard_loss' and not params.embedding_hidden_dim:
        ValueError('set the embedding_hidden_dim if triplet_semihard_loss is specified')
    _include_top = True
    _weights = None
    _metrics = ['categorical_accuracy', top_5_accuracy]
    if params.pretrained:
        logger.info('pretrained..')
        _include_top = False
        _weights = 'imagenet'
    # Define the top of the architecture
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
        x = base_model.output
        _inputs = base_model.input
        reshape_size = 1024
    elif params.model_architecture == 'resnet':
        base_model = tf.keras.applications.resnet50.ResNet50(input_shape=params.image_dim,
                                                             include_top=_include_top,
                                                             weights=_weights,
                                                             classes=params.nb_classes)
        x = base_model.output
        _inputs = base_model.input
        reshape_size = 2048
    elif params.model_architecture == 'densenet':
        base_model = tf.keras.applications.densenet.DenseNet121(input_shape=params.image_dim,
                                                                include_top=_include_top,
                                                                weights=_weights,
                                                                classes=params.nb_classes)
        x = base_model.output
        _inputs = base_model.input
        reshape_size = 1024
    elif params.model_architecture == 'convnet':
        " generic convnet"
        _inputs = layers.Input(shape=params.image_dim)
        filters = params.embedding_hidden_dim
        kernel = (3, 3)
        strides = (2, 2)
        x = _inputs
        for i in range(5):
            print(x)
            x = layers.Conv2D(filters, kernel,
                              padding='valid',
                              use_bias=False,
                              strides=strides,
                              name='conv{}'.format(i))(x)
            x = layers.BatchNormalization(axis=-1, name='conv{}_bn'.format(i))(x)
            x = layers.ReLU(6., name='conv{}_relu'.format(i))(x)

        reshape_size = filters
    else:
        raise ValueError("architecture not defined.")
    # If triplet loss, complete the structure
    if params.loss == 'triplet_semihard_loss':
        # re-set the metrics
        _metrics = None
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
        x = layers.Lambda(lambda _x: tf.keras.backend.l2_normalize(_x, axis=1),
                          name='conv_embedding_norm')(x)
        model = Model(inputs=_inputs, outputs=x)

        def _loss_fn(y_true, y_pred):
            y_true = tf.keras.backend.argmax(y_true, axis=-1)
            return triplet_semihard_loss(labels=y_true,
                                         embeddings=y_pred,
                                         margin=params.triplet_margin)
        _loss = _loss_fn
    # Complete the rest of the architecture if pretrained weights are loaded.
    # TODO: This should be marked as something like 'complete the top of the architechture
    elif params.pretrained:
        logger.info("append the top to the structure..")
        _loss = 'categorical_crossentropy'
        x = layers.GlobalAveragePooling2D()(x)
        shape = (1, 1, int(reshape_size * 1.0))
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(params.dropout, name='dropout')(x)
        x = layers.Conv2D(params.nb_classes, (1, 1),
                          padding='same',
                          name='conv_preds')(x)
        x = layers.Reshape((params.nb_classes,), name='reshape_2')(x)
        x = layers.Activation('softmax', name='act_softmax')(x)
        model = Model(inputs=_inputs, outputs=x)
    else:
        # If neither, entire structure is defined already.
        _loss = 'categorical_crossentropy'
        model = base_model

    if params.nb_layers_to_freeze:
        for i, layer in enumerate(model.layers):
            if i < params.nb_layers_to_freeze:
                layer.trainable = False
            else:
                logger.info(layer.name)
        logger.info("{} out of {} layers frozen..".format(params.nb_layers_to_freeze, i))

    if is_optimizer_adam:
        _opt = Adam(lr=params.lr_rate)
    else:
        _opt = sgd

    model.compile(optimizer=_opt,
                  loss=_loss,
                  metrics=_metrics
                  )
    tf.logging.info(model.summary())
    return model