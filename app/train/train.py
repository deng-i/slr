import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import RandomRotation
import argparse
from Rec_model import RecModel

       
def train(opt):
    input_size = (opt.row, opt.col, opt.ch)

    # Data Generator

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        opt.train_ad,  # this is the target directory
        target_size=(opt.row, opt.col),  # all images will be resized to 150x150
        batch_size=opt.batch_size,
        class_mode='categorical')  # since we use categorical_crossentropy loss, we need categorical labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        opt.validation_ad,
        target_size=(opt.row, opt.col),
        batch_size=opt.batch_size,
        class_mode='categorical')

    # check the name of each class with corresponding indices using:
    # print(train_generator.class_indices)
    # a, b = next(train_generator)
    # print(b)

    #####Compile####
    RecM = RecModel(input_size, opt.num_class)
    model = RecM.model_F
    _adam = optimizers.Adam(learning_rate=opt.lr)
    model.compile(loss='categorical_crossentropy', optimizer=_adam, metrics=['accuracy'])
    # print(model.summary())

    # model = tf.keras.models.load_model("/content/grdive/MyDrive/uni/project/hgr/3new_chpoint.33-0.99")

    model_checkpoint = ModelCheckpoint(opt.chekp, monitor='val_accuracy', verbose=1, save_best_only=False,
                                       save_weights_only=False)

    model.fit(
        train_generator,
        # steps_per_epoch=opt.num_img // opt.batch_size,
        epochs=opt.epochs,
        validation_data=validation_generator,
        # validation_steps=opt.num_val // opt.batch_size,
        callbacks=[model_checkpoint])


if __name__ == '__main__':
    class Option:
        def __init__(self):
            self.batch_size = 2


    opt = Option()
    opt.epochs = 150
    opt.lr = 0.001
    opt.input_size = (320, 320, 3)
    opt.train_ad = "../../../OUHANDS/train"
    opt.validation_ad = "../../../OUHANDS/test/val"
    opt.test_ad = "../../../OUHANDS/test"
    opt.chekp = "../models/chpoint.{epoch:02d}-{val_accuracy:.2f}"
    opt.model_path = "../chpoint5"
    opt.row = 320
    opt.col = 320
    opt.ch = 3
    opt.num_val = 33
    opt.num_img = 33
    opt.num_class = 10
    train(opt)
