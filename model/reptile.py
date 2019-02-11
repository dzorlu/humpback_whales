
from keras.models import load_model
import numpy as np

from model.model import create_model_fn
from data.generator import Generator
from data.triplet_generator import TripletGenerator

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NB_CLASSES_META_LEARNING = 10


def train_reptile(model_params, outer_step_size=0.02, steps=5):
    def fn(_gen):
        for a, b in _gen:
            yield a, b

    data_params = {
        'featurewise_center': False,
        'featurewise_std_normalization': False,
        'rotation_range': 20,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'zoom_range': 0.2,
        'shear_range': 0.4,
    }

    train_generator = TripletGenerator(file_path=model_params.file_path,
                                       image_path=model_params.image_train_path,
                                       image_test_path=model_params.image_test_path,
                                       nb_classes_batch=NB_CLASSES_META_LEARNING,
                                       nb_images_per_class_batch=5,
                                       validation_split=0.2,
                                       subset='training',
                                       is_reptile=True,
                                       **data_params)
    eval_generator = TripletGenerator(file_path=model_params.file_path,
                                      image_path=model_params.image_train_path,
                                      image_test_path=model_params.image_test_path,
                                      nb_classes_batch=NB_CLASSES_META_LEARNING,
                                      nb_images_per_class_batch=5,
                                      validation_split=0.2,
                                      subset='eval',
                                      is_reptile=True,
                                      **data_params)

    model = create_model_fn(model_params, is_optimizer_adam=False)

    nb_steps_per_epoch = len(train_generator)
    nb_steps_per_epoch_val = len(eval_generator)
    _generator = fn(train_generator)
    _eval_generator = fn(eval_generator)
    NB_EPOCHS = 1000
    # reptile training loop
    for i in range(NB_EPOCHS):
        print("epoch: {}".format(i))
        for _ in range(nb_steps_per_epoch):

            weights_before = model.weights

            # sample a new task. need to take k steps.
            x, y = next(_generator)
            # randomize bc generator does not
            ix = list(range(x.shape[0]))
            np.random.shuffle(ix)
            x, y = x[ix], y[ix]

            batch_size = x.shape[0] // steps
            # take k steps
            for s in range(steps):
                _x, _y = x[s * batch_size: (s + 1) * batch_size, :, :, :], y[s * batch_size: (s + 1) * batch_size, :]
                model.train_on_batch(_x, _y)
            weights_after = model.weights

            # meta update after task training.
            for w_ix in range(len(weights_before)):
                model.weights[i] = (weights_before[i] +
                                    (weights_after[i] - weights_before[i]) * outer_step_size)

        # eval
        outs_per_batch = []
        for s in range(nb_steps_per_epoch_val):
            x, y = next(_eval_generator)
            outs = model.test_on_batch(x, y)
            outs_per_batch.append(outs)
        averages = []
        # outs: (val_loss, accuracy, top_5_accuracy)
        stateful_metric_indices = [1, 2]
        for k in range(len(outs)):
            if k not in stateful_metric_indices:
                averages.append(np.average([out[k] for out in outs_per_batch]))
            else:
                # error on the last batch.
                averages.append(np.float64(outs_per_batch[-1][k]))
        print("epoch: {}, metrics: {}".format(i, averages))

    # TODO: eval metric. early stopping based on that.
    model.save(model_params.tmp_data_path + 'meta_learning_model.h5')
    return model, averages


# test time.
def inference_reptile(model_params, generator, test_generator, nn_classifier_predictions, nb_classes_meta=NB_CLASSES_META_LEARNING):
    """
    nn_classifier_predictions: [test_data_len, nb_classes]
    """
    x, img_names = generator.get_test_images_and_names()
    nn_classifier_predictions_class_ix = np.argsort(-nn_classifier_predictions)[:nb_classes_meta]
    assert len(x), len(nn_classifier_predictions_class_ix)
    # load the meta-training model
    model = load_model(model_params.tmp_data_path + 'meta_learning_model.h5')

    # each prediction becomes a task downstream
    ensemble_preds = []
    for i, task_class_ids in enumerate(nn_classifier_predictions_class_ix):
        weights_before = model.weights
        # training - training images
        batch_x, batch_y = test_generator.get_train_image_from_class_ix(task_class_ids)
        model.train_on_batch(batch_x, batch_y)
        # predict - test image
        outs = model.predict_on_batch(x[i])
        # take top 5 predictions
        pred_class_ix = np.argmax(-outs)[:5]
        ensemble_preds.append(pred_class_ix)
        # revert back for the next task
        model.weights = weights_before
    return np.vstack(ensemble_preds)

