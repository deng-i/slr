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

    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     opt.train_ad,
    #     image_size=(opt.row, opt.col),
    #     label_mode='categorical',
    #     batch_size=opt.batch_size
    # )
    #
    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #     opt.validation_ad,
    #     image_size=(opt.row, opt.col),
    #     label_mode='categorical',
    #     batch_size=opt.batch_size
    # )
    #
    # data_augmentation = tf.keras.Sequential([
    #     tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
    #     tf.keras.layers.experimental.preprocessing.RandomRotation(0.7),
    #     # tf.keras.layers.experimental.preprocessing.RandomShear(0.2),
    #     tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
    #     tf.keras.layers.experimental.preprocessing.RandomTranslation(0.15, 0.15)
    # ])
    #
    # rescale = tf.keras.layers.Rescaling(1. / 255)
    #
    # train_ds = train_ds.shuffle(200).batch(opt.batch_size).map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    # val_ds = val_ds.map(lambda x, y: (rescale(x), y))
    #
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_generator = train_ds.prefetch(buffer_size=AUTOTUNE)
    # validation_generator = val_ds.prefetch(buffer_size=AUTOTUNE)
    # train_generator = train_ds
    # validation_generator = val_ds
    # for a, b in train_generator:
    #     print(a.shape)
    #     print(b.shape)
    #     print(b)
    #     break

    #Data Generator

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
         )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        opt.train_ad,  # this is the target directory
        target_size=(opt.row, opt.col),  # all images will be resized to 150x150
        batch_size=opt.batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            opt.validation_ad,
            target_size=(opt.row, opt.col),
            batch_size=opt.batch_size,
            class_mode='categorical')

    # for a, b in train_generator:
    #     print(a.shape)
    #     print(b.shape)
    #     print(b)
    #     break

    test_generator = test_datagen.flow_from_directory(
        opt.test_ad,
        target_size=(opt.row, opt.col),
        batch_size=opt.batch_size,
        class_mode='categorical'
    )

    # check the name of each class with corresponding indices using:
    # train_generator.class_indices

    #####Compile####
    RecM = RecModel(input_size, opt.num_class)
    model = RecM.model_F
    _adam = optimizers.Adam(learning_rate=opt.lr, beta_1=0.9, beta_2=0.999)
    model.compile(loss='binary_crossentropy', optimizer=_adam, metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint(opt.chekp, monitor='val_accuracy', verbose=1, save_best_only=True)
    # print(model.summary())


    model.load_weights(opt.model_path)
    # model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    model.evaluate(test_generator)
    # model.fit(
    #         train_generator,
    #         # steps_per_epoch=opt.num_img // opt.batch_size,
    #         epochs=opt.epochs,
    #         validation_data=validation_generator,
    #         # validation_steps=opt.num_val // opt.batch_size,
    #         callbacks=[model_checkpoint])


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--batch_size', type=int, default=2)
    # # parser.add_argument('--input_size', type=list, default='(320,320,3)')
    # parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--lr', type=float, default=0.001)
    #
    # parser.add_argument('--train_ad', type=str, default='')
    # parser.add_argument('--validation_ad', type=str, default='')
    #
    # parser.add_argument('--chekp', type=str, default='')
    # parser.add_argument('--row', type=int, default=320)
    # parser.add_argument('--col', type=int, default=320)
    # parser.add_argument('--ch', type=int, default=3)
    # parser.add_argument('--num_val', type=int, default=3)
    # parser.add_argument('--num_img', type=int, default=3)
    # parser.add_argument('--num_class', type=int, default=3)
    #
    #
    #
    # opt = parser.parse_args()
    class Option:
        def __init__(self):
            self.batch_size = 2


    opt = Option()
    opt.epochs = 150
    opt.lr = 0.001
    opt.input_size = (320, 320, 3)
    opt.train_ad = "../../OUHANDS/train/train"
    opt.validation_ad = "../../OUHANDS/train/val"
    opt.test_ad = "../../OUHANDS/test"
    opt.chekp = "chpoint"
    opt.model_path = "../3chpoint.11-1.00.hdf5"
    opt.row = 480
    opt.col = 320
    opt.ch = 3
    opt.num_val = 33
    opt.num_img = 33
    opt.num_class = 10
    train(opt)
