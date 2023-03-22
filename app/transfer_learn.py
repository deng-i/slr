import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from train.Rec_model import RecModel
import os
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class Transfer:
    def __init__(self, model_path: str, training_data_path: str):
        self.training_data_path = training_data_path
        self.split_data()
        self.load_data()
        # model = tf.keras.models.load_model(model_path, compile=False)
        # for layer in model.layers[:-1]:
        #     layer.trainable = False
        # print(model.summary())
        RecM = RecModel((320, 320, 3), len(self.training_data.class_indices))
        model = RecM.model_F
        # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        prediction = tf.keras.layers.Dense(len(self.training_data.class_indices),
                                           activation="softmax")(model.layers[-2].output)
        self.transfer_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        # print(self.transfer_model.summary())

    def train(self):
        adam = tf.keras.optimizers.Adam()
        self.transfer_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model_checkpoint = ModelCheckpoint("models/trained_model", monitor='val_accuracy', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor="val_accuracy", patience=5)
        self.transfer_model.fit(
            self.training_data,
            epochs=40,
            validation_data=self.val_data,
            callbacks=[model_checkpoint, early_stop]
        )
        self.transfer_model.save("models/trained_model_end")

    def load_data(self):
        train_datagen = ImageDataGenerator(
            rotation_range=40,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest'
        )

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.training_data = train_datagen.flow_from_directory(
            os.path.join(self.training_data_path, "train"),  # this is the target directory
            target_size=(320, 320),  # all images will be resized to 320*320
            batch_size=2,
            class_mode='categorical')

        self.val_data = test_datagen.flow_from_directory(
            os.path.join(self.training_data_path, "val"),  # this is the target directory
            target_size=(320, 320),  # all images will be resized to 320*320
            batch_size=2,
            class_mode='categorical')

    def split_data(self):
        """
        Splits the data into training and validation
        """
        data_dir = self.training_data_path
        train_data_dir = os.path.join(os.path.dirname(data_dir), "train")
        self.training_data_path = train_data_dir
        # reset the data
        if os.path.exists(os.path.join(train_data_dir)):
            shutil.rmtree(train_data_dir)

        # create directories
        os.mkdir(train_data_dir)
        train_dir = os.path.join(train_data_dir, 'train')
        os.mkdir(train_dir)
        val_dir = os.path.join(train_data_dir, 'val')
        os.mkdir(val_dir)

        val_split = 0.2

        # get a list of all the sign names
        sign_names = os.listdir(data_dir)

        for sign in sign_names:
            train_path = os.path.join(train_dir, sign)
            val_path = os.path.join(val_dir, sign)
            os.mkdir(train_path)
            os.mkdir(val_path)
            signs_path = os.path.join(data_dir, sign)
            signs = os.listdir(signs_path)
            train_filenames, val_filenames = train_test_split(signs, test_size=val_split)

            # move the train files
            for filename in train_filenames:
                src = os.path.join(signs_path, filename)
                dst = os.path.join(train_path, filename)
                shutil.copy(src, dst)

            # move the val files
            for filename in val_filenames:
                src = os.path.join(signs_path, filename)
                dst = os.path.join(val_path, filename)
                shutil.copy(src, dst)


