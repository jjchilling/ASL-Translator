"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import VGGModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np
from live import live
import skimage.segmentation as seg
import copy
from sklearn.utils import check_random_state

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--task',
        choices=['1', '3'],
        help='''Which task of the assignment to run -
        training from scratch (1), or fine tuning VGG-16 (3).''')
    parser.add_argument(
        '--data',
        default='..'+os.sep+'data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--load-vgg',
        default='vgg16_imagenet.h5',
        help='''Path to pre-trained VGG-16 file (only applicable to
        task 3).''')
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')
    parser.add_argument(
        '--lime-image',
        default='test/Bedroom/image_0003.jpg',
        help='''Name of an image in the dataset to use for LIME evaluation.''')

    return parser.parse_args()


def LIME_explainer(model, path, preprocess_fn, timestamp):
    """
    This function takes in a trained model and a path to an image and outputs 4
    visual explanations using the LIME model
    """

    save_directory = "lime_explainer_images" + os.sep + timestamp
    if not os.path.exists("lime_explainer_images"):
        os.mkdir("lime_explainer_images")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    image_index = 0

    print(save_directory)

    # def image_and_mask(title, positive_only=True, num_features=5,
    #                    hide_rest=True):
    #     nonlocal image_index

    #     temp, mask = explanation.get_image_and_mask(
    #         explanation.top_labels[0], positive_only=positive_only,
    #         num_features=num_features, hide_rest=hide_rest)
    #     # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    #     plt.title(title)

    #     image_save_path = save_directory + os.sep + str(image_index) + ".png"
    #     plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    #     plt.show()

    #     image_index += 1

    # Read the image and preprocess it as before
    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
    image = preprocess_fn(image)

    
    # explainer = lime_image.LimeImageExplainer()


    # segments = seg.felzenszwalb(image)
    # fudged_image = image.copy()
    # for x in np.unique(segments):
    #     fudged_image[segments == x] = (
    #         np.mean(image[segments == x][:, 0]),
    #         np.mean(image[segments == x][:, 1]),
    #         np.mean(image[segments == x][:, 2]))

    # # labels = explainer.data_labels(image, None, None, model.predict, 1000)
    # # data, labels = explainer.data_labels(image, fudged_image, segments,
    # #                                     model.predict, 1000)
    # random_state = check_random_state(0)
    # n_features = np.unique(segments).shape[0]
    # data = random_state.randint(0, 2, 1000 * n_features).reshape((1000, n_features))


    # labels = []
    # data[0, :] = 1
    # imgs = []
    # for row in (data):
    #     temp = copy.deepcopy(image)
    #     zeros = np.where(row == 0)[0]
    #     mask = np.zeros(segments.shape).astype(bool)
    #     for z in zeros:
    #         mask[segments == z] = True
    #     temp[mask] = fudged_image[mask]
    #     imgs.append(temp)
    #     if len(imgs) == 10:
    #         preds = model.predict(np.array(imgs))
    #         labels.extend(preds)
    #         imgs = []
    # if len(imgs) > 0:
    #     preds = model.predict(np.array(imgs))
    #     labels.extend(preds)
    
    # # preds = model.predict(np.array(image))
    # # print("preds: ", preds)
    # print("PREDS", preds)
    # print("LABELS", labels)
    # print("DATA", data)

    # model.predict(image)

    # Datasets.get_data(dataset, "../data/train/", True, True, True)

    # data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    # data_gen = data_gen.flow_from_directory(
    #         path,
    #         target_size=(image.shape[0], image.shape[1]),
    #         class_mode='sparse',
    #         batch_size=hp.batch_size,
    #         shuffle=None,
    #         classes=None)
    # model.predict(data_gen)

    # explanation = explainer.explain_instance(
    #     image.astype('double'), model.predict, top_labels=5, hide_color=0,
    #     num_samples=1000)

    # # The top 5 superpixels that are most positive towards the class with the
    # # rest of the image hidden
    # image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
    #                hide_rest=True)

    # # The top 5 superpixels with the rest of the image present
    # image_and_mask("Top 5 with the rest of the image present",
    #                positive_only=True, num_features=5, hide_rest=False)

    # # The 'pros and cons' (pros in green, cons in red)
    # image_and_mask("Pros(green) and Cons(red)",
    #                positive_only=False, num_features=10, hide_rest=False)

    # # Select the same class explained on the figures above.
    # ind = explanation.top_labels[0]
    # # Map each explanation weight to the corresponding superpixel
    # dict_heatmap = dict(explanation.local_exp[ind])
    # heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    # plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    # plt.colorbar()
    # plt.title("Map each explanation weight to the corresponding superpixel")

    # image_save_path = save_directory + os.sep + str(image_index) + ".png"
    # plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    # plt.show()


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, ARGS.task, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,            # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():

    # loading model
    datasets = Datasets(ARGS.data, ARGS.task)
    model = VGGModel()
    # checkpoint_path = "checkpoints" + os.sep + "vgg_model" + os.sep + timestamp + os.sep
    # logs_path = "logs" + os.sep + "vgg_model" + os.sep + timestamp + os.sep
    model(tf.keras.Input(shape=(224, 224, 3)))
    # model.vgg16.summary()
    # model.head.summary()
    model.vgg16.load_weights('vgg16_imagenet.h5', by_name=True)
    # model.vgg16.load_weights(ARGS.load_vgg, by_name=True)
    model.head.load_weights('checkpoints/vgg_model/042324-233248/vgg.weights.e026-acc0.9286.h5', by_name=False)
    path = '../data/test/A/A_test.jpg'
    timestamp = datetime.now().strftime("%m%d%y-%H%M%S")
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])
    # test(model, datasets.test_data)


    image = imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = resize(image, (hp.img_size, hp.img_size, 3), preserve_range=True)
    image = datasets.preprocess_fn(image)
    test = datasets.get_data("../data/test/", True, True, True)


    """ a bunch of attempts in LIME_explainer """
    # LIME_explainer(model, path, datasets.preprocess_fn, timestamp)



    """ trying to just use model()"""
    # probabilities = model(np.array([image])).numpy()[0]
    # predict_class_idx = np.argmax(probabilities)
    # print(predict_class_idx)

    """ BEST VERSION I THINK trying to just use model() how tensorboard does"""
    # can replace datasets.test_data with test to check the folder with one image
    for batch in datasets.test_data:
        for i, image in enumerate(batch[0]):
            correct_class_idx = batch[1][i]
            probabilities = model(np.array([image])).numpy()[0]
            predict_class_idx = np.argmax(probabilities)
            print(correct_class_idx, predict_class_idx)

    """ using model.predict() """""
    # first for test (from "one" folder)
    # test_pred = model.predict(test[0])
    # test_true = test[0][1]
    # test_true = np.array(test_true).flatten()
    # test_pred = np.array(test_pred)
    # test_pred = np.argmax(test_pred, axis=-1).flatten()
    # print("TEST PRED: ", test_pred)
    # print("TEST TRUE: ", test_true)

    # then for previous test data
    # real_pred = model.predict(datasets.test_data[0][0])
    # real_true = datasets.test_data[0][1]
    # test_true = np.array(real_true).flatten()
    # real_pred = np.array(real_pred)
    # real_pred = np.argmax(real_pred, axis=-1).flatten()
    # print("real PRED: ", real_pred)
    # print("real TRUE: ", real_true)
    # print("CLASSES: ", datasets.classes)


    # """ Main function. """

    # time_now = datetime.now()
    # timestamp = time_now.strftime("%m%d%y-%H%M%S")
    # init_epoch = 0

    # # If loading from a checkpoint, the loaded checkpoint's directory
    # # will be used for future checkpoints
    # if ARGS.load_checkpoint is not None:
    #     ARGS.load_checkpoint = os.path.abspath(ARGS.load_checkpoint)

    #     # Get timestamp and epoch from filename
    #     regex = r"(?:.+)(?:\.e)(\d+)(?:.+)(?:.h5)"
    #     init_epoch = int(re.match(regex, ARGS.load_checkpoint).group(1)) + 1
    #     timestamp = os.path.basename(os.path.dirname(ARGS.load_checkpoint))

    # # If paths provided by program arguments are accurate, then this will
    # # ensure they are used. If not, these directories/files will be
    # # set relative to the directory of main.py
    # if os.path.exists(ARGS.data):
    #     ARGS.data = os.path.abspath(ARGS.data)
    # if os.path.exists(ARGS.load_vgg):
    #     ARGS.load_vgg = os.path.abspath(ARGS.load_vgg)

    # # Run script from location of main.py
    # os.chdir(sys.path[0])

    # datasets = Datasets(ARGS.data, ARGS.task)

    # model = VGGModel()
    # checkpoint_path = "checkpoints" + os.sep + \
    #         "vgg_model" + os.sep + timestamp + os.sep
    # logs_path = "logs" + os.sep + "vgg_model" + \
    #         os.sep + timestamp + os.sep
    # model(tf.keras.Input(shape=(224, 224, 3)))

    #     # Print summaries for both parts of the model
    # model.vgg16.summary()
    # model.head.summary()

    #     # Load base of VGG model
    # model.vgg16.load_weights(ARGS.load_vgg, by_name=True)

    # # Load checkpoints
    # if ARGS.load_checkpoint is not None:
    #     model.head.load_weights(ARGS.load_checkpoint, by_name=False)

    # # Make checkpoint directory if needed
    # if not ARGS.evaluate and not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)

    # # Compile model graph
    # model.compile(
    #     optimizer=model.optimizer,
    #     loss=model.loss_fn,
    #     metrics=["sparse_categorical_accuracy"])

    # if ARGS.evaluate:
    #     test(model, datasets.test_data)

    #     path = ARGS.lime_image
    #     LIME_explainer(model, path, datasets.preprocess_fn, timestamp)
    # else:
    #     train(model, datasets, checkpoint_path, logs_path, init_epoch)

# Make arguments global
ARGS = parse_args()
# live()
main()
