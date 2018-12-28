
import sys
import argparse
import datetime
import os

import tensorflow as tf
from tensorflow.keras.callbacks import  ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import callbacks
from tensorflow.python.keras import backend as K
import numpy as np
import pandas as pd


from model.model import create_model_fn
from data.generator import Generator
from data.triplet_generator import TripletGenerator


class LearningRateRangeTest(callbacks.Callback):
    def __init__(self, total_nb_steps, base_rate=10e-5, max_rate=10e0):
        self.max_rate = np.log10(max_rate)
        self.base_rate = np.log10(base_rate)
        self.total_nb_steps = float(total_nb_steps)
        self.steps_taken = 0
        super(LearningRateRangeTest, self).__init__()

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, 10 ** self.base_rate)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.steps_taken += 1
        _lr = 10 ** (self.base_rate * (1 - self.steps_taken / self.total_nb_steps))
        logs['lr'] = np.float64(_lr)
        K.set_value(self.model.optimizer.lr, np.float64(_lr))


class CosineLearninRatePolicy(callbacks.Callback):
    def __init__(self, max_rate, total_nb_steps):
        self.base_rate = max_rate / 10
        self.max_rate = max_rate
        self.steps_taken = 0
        self.cycle_length = float(total_nb_steps) / 2 # nb steps per epoch / 2
        super(CosineLearninRatePolicy, self).__init__()

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.max_rate)
        # for tensorboard
        logs['lr'] = np.float64(self.max_rate)

    def on_batch_end(self, batch, logs=None):
        self.steps_taken += 1
        _scaler = (1 + np.cos(np.pi * self.steps_taken / self.cycle_length)) / 2
        _lr = self.base_rate + (self.max_rate - self.base_rate) * _scaler
        # for tensorboard
        logs['lr'] = np.float64(_lr)
        K.set_value(self.model.optimizer.lr, _lr)


def create_submission(generator, model, model_params, nn_classifier=None):
    x, img_names = generator.get_test_images_and_names()
    # no shuffling ensures order
    test_generator = generator.get_test_generator(x)
    preds = model.predict_generator(test_generator)
    _path = model_params.tmp_data_path + 'preds_test.npy'
    np.save(_path, preds)
    preds_out = []
    if model_params.loss == 'triplet_semihard_loss':
        # postprocessing step with NN classifier.
        preds = nn_classifier.predict_proba(preds)
    for c_pred in preds:
        c_pred = np.argsort(-1 * c_pred)[:4]
        # this always appends 'new whale'.
        preds_out.append([generator.class_inv_indices[p] for p in c_pred]+['new_whale'])
    print(len(preds_out))
    print(len(img_names))
    preds = [' '.join([col for col in row]) for row in preds_out]
    submission = pd.DataFrame(np.array([img_names, preds]).T, columns=['Image', 'Id'])
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filepath = os.path.join(model_params.tmp_data_path, "submission_{}_{}.csv".format(model_params.model_architecture, ts))
    submission.to_csv(filepath, index=False)
    print("{} predictions persisted..".format(len(submission)))


def get_callbacks(model_params, patience=3):
    weight_path = model_params.tmp_data_path + "{}_{}_weights.best.hdf5".format(model_params.loss, model_params.model_architecture)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', period=1)

    early = EarlyStopping(monitor="val_loss",
                          min_delta=0.0,
                          mode="min",
                          patience=patience*5)

    embeddings_freq = 0
    embeddings_layer_names = None
    if model_params.loss == 'triplet_semihard_loss':
        embeddings_freq = 1
        embeddings_layer_names = ['conv_embedding_norm']

    tensorboard = callbacks.TensorBoard(log_dir=model_params.tmp_data_path,
                                        histogram_freq=0,
                                        batch_size=model_params.batch_size,
                                        write_graph=True, write_grads=False,
                                        write_images=False,
                                        embeddings_layer_names=embeddings_layer_names,
                                        embeddings_metadata=None, embeddings_data=None)
    if model_params.lr_policy == 'range_test':
        lr_policy = LearningRateRangeTest(total_nb_steps=model_params.total_nb_steps * 10)
    elif model_params.lr_policy == 'cosine_rate_policy':
        lr_policy = CosineLearninRatePolicy(total_nb_steps=model_params.total_nb_steps * 10,
                                            max_rate=model_params.lr_rate)
    else:
        min_lr_rate = model_params.lr_rate / 10
        lr_policy = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience,
                                      verbose=1, mode='auto', cooldown=3, min_lr=min_lr_rate)
    callbacks_list = [checkpoint, early]
    callbacks_list += [lr_policy]
    # tensorboard evaluated last bc it reads appended logs dict
    callbacks_list += [tensorboard]

    return callbacks_list, weight_path


def main(args):
    model_params = tf.contrib.training.HParams(
        fit=FLAGS.fit,
        file_path=FLAGS.file_path,
        image_train_path=FLAGS.image_train_path,
        image_test_path=FLAGS.image_test_path,
        model_architecture=FLAGS.model_architecture,
        tmp_data_path=FLAGS.tmp_data_path,
        batch_size=FLAGS.batch_size,
        nb_epochs=FLAGS.nb_epochs,
        lr_rate=FLAGS.lr_rate,
        lr_policy=FLAGS.lr_policy,
        nb_layers_to_freeze=FLAGS.nb_layers_to_freeze,
        pretrained=FLAGS.pretrained,
        loss=FLAGS.loss,
        embedding_hidden_dim=FLAGS.embedding_hidden_dim,
        triplet_margin=FLAGS.triplet_margin,
        dropout=FLAGS.dropout)
    print(model_params)
    #
    data_params = {
        'featurewise_center': False,
        'featurewise_std_normalization': False,
        'rotation_range': 20,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.3,
        'shear_range': 0.4,
       }

    # fetch different generator based on loss type
    if model_params.loss != 'triplet_semihard_loss':
        gen = Generator(file_path=model_params.file_path,
                        image_path=model_params.image_train_path,
                        image_test_path=model_params.image_test_path,
                        batch_size=model_params.batch_size,
                        **data_params)
        model_params.add_hparam('nb_classes', gen.get_nb_classes())
        model_params.add_hparam('image_dim', gen.target_size)
        model_params.add_hparam('class_weights', gen.get_class_weights())
        model_params.add_hparam('total_nb_steps', len(gen.get_train_generator()))
        train_generator = gen.get_train_generator()
        eval_generator = gen.get_eval_generator()
    else:
        train_generator = TripletGenerator(file_path=model_params.file_path,
                                           image_path=model_params.image_train_path,
                                           image_test_path=model_params.image_test_path,
                                           nb_classes_batch=16,
                                           nb_images_per_class_batch=4,
                                           validation_split=0.2,
                                           subset='training',
                                           **data_params)
        eval_generator = TripletGenerator(file_path=model_params.file_path,
                                          image_path=model_params.image_train_path,
                                          image_test_path=model_params.image_test_path,
                                          nb_classes_batch=16,
                                          nb_images_per_class_batch=4,
                                          validation_split=0.2,
                                          subset='eval',
                                          **data_params)
        model_params.add_hparam('nb_classes', train_generator.get_nb_classes())
        model_params.add_hparam('image_dim', train_generator.target_size)
        model_params.add_hparam('class_weights', None)
        model_params.add_hparam('total_nb_steps', len(train_generator))

    _callbacks, weight_path = get_callbacks(model_params)
    model = create_model_fn(model_params)

    if model_params.fit:
        model.fit_generator(generator=train_generator,
                            validation_data=eval_generator,
                            use_multiprocessing=True,
                            epochs=model_params.nb_epochs,
                            callbacks=_callbacks)
    # evaluate
    model.load_weights(weight_path)
    eval_res = model.evaluate_generator(eval_generator)
    print('eval {}'.format(eval_res))

    neigh = None
    if model_params.loss == 'triplet_semihard_loss':
        from sklearn.neighbors import KNeighborsClassifier
        nb_neighbors = 3
        neigh = KNeighborsClassifier(nb_neighbors)
        # need to train a NN classifier with embeddings for inference at test time.
        # no augmentation
        gen = Generator(file_path=model_params.file_path,
                        image_path=model_params.image_train_path,
                        image_test_path=model_params.image_test_path,
                        batch_size=model_params.batch_size)
        classes = list(gen.classes)
        # this need to line up with labels. hence no shuffle.
        _generator = gen.get_train_generator(shuffle=False)
        predictions = list(model.predict_generator(_generator))
        # augment on
        gen = Generator(file_path=model_params.file_path,
                        image_path=model_params.image_train_path,
                        image_test_path=model_params.image_test_path,
                        batch_size=model_params.batch_size,
                        **data_params)
        classes += list(gen.classes)
        # this need to line up with labels. hence no shuffle.
        _generator = gen.get_train_generator(shuffle=False)
        predictions += list(model.predict_generator(_generator))
        # fit
        neigh.fit(predictions, classes)
        # save
        _path = model_params.tmp_data_path + 'preds_train.npy'
        np.save(_path, np.array(predictions))
        _path = model_params.tmp_data_path + 'labels_train.npy'
        np.save(_path, np.array(classes))
        import pickle
        _path = model_params.tmp_data_path + 'model.pcl'
        pickle.dump(neigh, open(_path, 'wb'))

    # test
    create_submission(gen, model, model_params, neigh)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--fit",
      type=bool,
      default=True,
      help="Hidden dim")
  parser.add_argument(
      "--loss",
      type=str,
      choices=['categorical_crossentropy', 'triplet_semihard_loss'],
      default="triplet_semihard_loss",
      help="Loss type / Model")
  parser.add_argument(
      "--embedding_hidden_dim",
      type=int,
      default=128,
      help="Hidden dim")
  parser.add_argument(
      "--triplet_margin",
      type=float,
      default=1.0,
      help="Triplet loss margin")
  parser.add_argument(
      "--file_path",
      type=str,
      default="",
      help="Path to training data dataframe")
  parser.add_argument(
      "--image_train_path",
      type=str,
      default="",
      help="Path to train images")
  parser.add_argument(
      "--image_test_path",
      type=str,
      default="",
      help="Path to test images)")
  parser.add_argument(
      "--tmp_data_path",
      type=str,
      default="/tmp/whales/",
      help="Path to temp data")
  parser.add_argument(
      "--dropout",
      type=float,
      default=0.5,
      help="Dropout used for convolutions and bidi lstm layers.")
  parser.add_argument(
      "--batch_size",
      type=int,
      default=128,
      help="Batch size to use for training/evaluation. Not valid for one-shot-learning")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="/tmp/whales",
      help="Path for storing the model checkpoints.")
  parser.add_argument(
      "--model_architecture",
      type=str,
      default="mobilenet",
      help="Model architecture.")
  parser.add_argument(
      "--nb_epochs",
      type=int,
      default=50,
      help="number of epochs")
  parser.add_argument(
      "--lr_policy",
      choices=['cosine_rate_policy', 'range_test', 'reduce'],
      type=str,
      default='reduce',
      help="lr policy")
  parser.add_argument(
      "--lr_rate",
      type=float,
      default=5e-4)
  parser.add_argument(
      "--nb_layers_to_freeze",
      type=int,
      default=0)
  parser.add_argument(
      "--pretrained",
      type=bool,
      default=False
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

