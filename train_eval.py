
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
        logs['lr'] = np.float64(self.max_rate)

    def on_batch_end(self, batch, logs=None):
        self.steps_taken += 1
        _scaler = (1 + np.cos(np.pi * self.steps_taken / self.cycle_length)) / 2
        _lr = self.base_rate + (self.max_rate - self.base_rate) * _scaler
        logs['lr'] = np.float64(_lr)
        K.set_value(self.model.optimizer.lr, _lr)


def create_submission(generator, model, model_params):
    x, img_names = generator.get_test_images_and_names()
    test_generator = generator.get_test_generator(x)
    preds = model.predict_generator(test_generator)
    preds_out = []
    for c_pred in preds:
        c_pred = np.argsort(-1 * c_pred)[:4]
        preds_out.append([generator.class_inv_indices[p] for p in c_pred])
        preds_out.append(['new_whale'])
    print(len(preds_out))
    print(len(img_names))
    preds = [' '.join([col for col in row]) for row in preds_out]
    submission = pd.DataFrame(np.array([img_names, preds]).T, columns=['Image', 'Id'])
    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filepath = os.path.join(model_params.tmp_data_path, "submission_{}_{}.csv".format(model_params.model_architecture, ts))
    submission.to_csv(filepath, index=False)
    print("{} predictions persisted..".format(len(submission)))


def get_callbacks(model_params, patience=2):
    weight_path = "{}/{}_weights.best.hdf5".format(model_params.tmp_data_path, model_params.model_architecture)

    checkpoint = ModelCheckpoint(weight_path, monitor='loss', verbose=1,
                                 save_best_only=True, mode='min', period=1)

    early = EarlyStopping(monitor="loss",
                          min_delta=0.0,
                          mode="min",
                          patience=patience*3)

    tensorboard = callbacks.TensorBoard(log_dir=model_params.tmp_data_path,
                                        histogram_freq=0,
                                        batch_size=model_params.batch_size,
                                        write_graph=True, write_grads=False,
                                        write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None,
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
    #callbacks_list = [early]
    callbacks_list = [checkpoint, early]
    callbacks_list += [lr_policy]
    callbacks_list += [tensorboard]

    return callbacks_list, weight_path


def main(args):
    model_params = tf.contrib.training.HParams(
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
        dropout=FLAGS.dropout)
    print(model_params)
    #
    data_params = {
        'featurewise_center': False,
        'featurewise_std_normalization': False,
        'rotation_range': 20,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.2,
        'shear_range': 0.4,
       }
    gen = Generator(file_path=model_params.file_path,
                    image_path=model_params.image_train_path,
                    image_test_path=model_params.image_test_path,
                    batch_size=model_params.batch_size,
                    **data_params)
    model_params.add_hparam('nb_classes', gen.get_nb_classes())
    model_params.add_hparam('image_dim', gen.target_size)
    model_params.add_hparam('class_weights', gen.get_class_weights())
    model_params.add_hparam('total_nb_steps', len(gen.get_train_generator()))

    model = create_model_fn(model_params)
    _callbacks, weight_path = get_callbacks(model_params)
    train_generator = gen.get_train_generator()
    #eval_generator = gen.get_eval_generator()

    model.fit_generator(generator=train_generator,
                        #validation_data=eval_generator,
                        #class_weight=model_params.class_weights,
                        epochs=model_params.nb_epochs,
                        callbacks=_callbacks)
    # evaluate
    model.load_weights(weight_path)
    # eval_res = model.evaluate_generator(eval_generator)
    # print('Accuracy: %2.1f%%, Top 3 Accuracy %2.1f%%' % (100 * eval_res[1], 100 * eval_res[2]))

    # test
    create_submission(gen, model, model_params)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--file_path",
      type=str,
      default="",
      help="Path to training/eval/test data")
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
      default="/tmp/whales",
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
      help="Batch size to use for training/evaluation.")
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
      default='cosine_rate_policy',
      help="lr policy")
  parser.add_argument(
      "--lr_rate",
      type=float,
      default=5e-3)
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

