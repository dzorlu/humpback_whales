

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def top_5_accuracy(x, y): return top_k_categorical_accuracy(x, y, 5)


def create_model_fn(params):
    """
    create model function and callbacks given the params
    :return:
    """
    if params.image_dim[0] not in [224, 128]:
        ValueError('hip to be square..')
    if params.nb_layers_to_freeze and not params.pretrained:
        ValueError('set the pretrained to TRUE if nb_layers_to_freeze is specified')
    _include_top = True
    _weights = None
    if params.pretrained:
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
    elif params.model_architecture == 'resnet':
        base_model = tf.keras.applications.resnet50.ResNet50(input_shape=params.image_dim,
                                                             include_top=_include_top,
                                                             weights=_weights,
                                                             classes=params.nb_classes)
    else:
        raise ValueError("architecture not defined.")
    if params.pretrained:
        x = base_model.output
        shape = (1, 1, int(1024 * 1.0))
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape(shape, name='reshape_1')(x)
        x = layers.Dropout(params.dropout, name='dropout')(x)
        x = layers.Conv2D(params.nb_classes, (1, 1),
                          padding='same',
                          name='conv_preds')(x)
        x = layers.Activation('softmax', name='act_softmax')(x)
        x = layers.Reshape((params.nb_classes,), name='reshape_2')(x)
        model = Model(inputs=base_model.input, output=x)
    else:
        model = base_model
    if params.nb_layers_to_freeze:
        for i, layer in enumerate(model.layers):
            if i < params.nb_layers_to_freeze:
                layer.trainable = False
        print("{} out of {} layers frozen..".format(params.nb_layers_to_freeze, i))
        model.compile(optimizer=Adam(lr=params.lr_rate),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', top_5_accuracy])
    tf.logging.info(base_model.summary())
    return model