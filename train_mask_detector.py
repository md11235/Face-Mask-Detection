# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Softmax, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from layers import ChannelWiseDotProduct, FusedFeatureSpectrum
from keras.backend.tensorflow_backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

tf.keras.backend.set_image_data_format('channels_last')


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
ap.add_argument("-r", "--resume", type=str, default=None,
                help="path to previously trained model")
ap.add_argument("-e", "--numepoch", type=int, default=0,
                help="start epoch number")
ap.add_argument("-l", "--learningrate", type=float, default=0.0,
                help="learning rate")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 200000
BS = 32

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	if not label in ['AroundNeck', 'Correct', 'BelowNose', 'NoMask']:
	# if not label in ['with_mask', 'without_mask']:
	    continue

	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

num_classes = len(set(labels))

# perform one-hot encoding on the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# labels = to_categorical(labels)

encoder = OneHotEncoder(sparse=False)
# transform data
labels = labels.reshape((-1, 1))
labels = encoder.fit_transform(labels)

# print(encoder.inverse_transform(np.asarray([[0., 0., 0., 1.]])))
# print(encoder.inverse_transform(np.asarray([[0., 0., 1., 0.]])))
# print(encoder.inverse_transform(np.asarray([[0., 1., 0., 0.]])))
# print(encoder.inverse_transform(np.asarray([[1., 0., 0., 0.]])))

# exit()
print(labels)
print(labels.shape)

#print(encoder.categories_)

#exit()

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(
    data, labels,
	test_size=0.20, stratify=labels, random_state=1337)

# print(np.unique(trainY, axis=0, return_counts=True))
# print(np.unique(testY, axis=0, return_counts=True))

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

if args["resume"] is None:
    print("Train from scratch.")
    # load the MobileNetV2 network, ensuring the head FC layer sets are
    # left off
    baseModel = ResNet152V2(weights="imagenet", include_top=False,
    	input_tensor=Input(shape=(224, 224, 3)))
    
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel_orig = baseModel.output

    headModel_gap = AveragePooling2D(pool_size=(7, 7))(headModel_orig)
    attention_branch = Conv2D(2048, 3, padding='same')(headModel_gap)
    attention_branch = Conv2D(2048, 1, padding='same')(attention_branch)
    attention_branch = Softmax()(attention_branch)
    # attention_branch = ChannelWiseDotProduct()(attention_branch, headModel_orig)
    # attention_branch = FusedFeatureSpectrum()(attention_branch, headModel_orig)
    attention_branch = ChannelWiseDotProduct()(attention_branch, headModel_gap)
    attention_branch = FusedFeatureSpectrum()(attention_branch, headModel_gap)
    flat_attention = Flatten(name="flatten")(attention_branch)

    headModel_gap_flattern = Flatten(name="headModel_gap_flattern")(headModel_gap)
    headModel = Concatenate()([headModel_gap_flattern, flat_attention])

    headModel = Dense(256, activation="relu")(headModel)
    headModel = Dropout(0.1)(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dense(num_classes, activation="softmax")(headModel)
    
    # headModel = AveragePooling2D(pool_size=(7, 7))(headModel_orig)
    # headModel = Flatten(name="flatten")(headModel)
    # headModel = Dense(256, activation="relu")(flat_attention)
    # headModel = Dropout(0.1)(headModel)
    # headModel = Dense(128, activation="relu")(headModel)
    # headModel = Dense(num_classes, activation="softmax")(headModel)
    
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    model.summary()

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
    	layer.trainable = True
    
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=1.0)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
    	metrics=["accuracy"])
    start_epoch = 0
else:
    print("Resume training at {}".format(args["resume"]))
    start_epoch = args["numepoch"]
    model = load_model(args["resume"])

    if args["learningrate"] > 0.0:
        K.set_value(model.optimizer.lr, args["learningrate"])

# train the head of the network
print("[INFO] training head...")


def scheduler(epoch):
    if epoch%100==0 and epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        new_lr = lr*0.999
        K.set_value(model.optimizer.lr, new_lr)
        print("lr changed to {}".format(new_lr))
    else: 
        print("lr remains {}".format(K.get_value(model.optimizer.lr)))

    return K.get_value(model.optimizer.lr)


lr_schedule = LearningRateScheduler(scheduler)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=300)
model_checkpoint =  ModelCheckpoint("trained_models/" + 'mobilenet_v2_face_epoch_{epoch:09d}_loss{val_loss:.4f}.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1)
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	initial_epoch=start_epoch,
	epochs=EPOCHS,
	callbacks=[early_stopping, model_checkpoint, lr_schedule]
)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

print(predIdxs)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
predY = tf.one_hot(predIdxs, depth=num_classes)
print(predY)
# show a nicely formatted confusion matrix
# cat_names = encoder.categories_[0]

# cmtx = pd.DataFrame(
#     confusion_matrix([cat_names[ind] for ind in testY.argmax(axis=1)],
#                      [cat_names[ind] for ind in predIdxs],
#                      labels=cat_names),
#     index=["true: {}".format(cn) for cn in cat_names],
#     columns=["pred: {}".format(cn) for cn in cat_names]
# )

# print(cmtx)

testY_labels = encoder.inverse_transform(testY)
predY_labels = encoder.inverse_transform(predY)

# print(testY_labels)
# print(predY_labels)
# print(testY_labels.shape)
# print(predY_labels.shape)
y_true = pd.Series(testY_labels.reshape((-1,)))
y_pred = pd.Series(predY_labels.reshape((-1,)))

print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(args["plot"])
