import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from train.Rec_model import RecModel


class Transfer:
    def __init__(self, model_path, training_data_path):
        self.training_data_path = training_data_path
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
        print(self.transfer_model.summary())

    def train(self):
        adam = tf.keras.optimizers.Adam()
        self.transfer_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.transfer_model.fit(
            self.training_data,
            epochs=40
        )
        self.transfer_model.save("trained_model")

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

        self.training_data = train_datagen.flow_from_directory(
            self.training_data_path,  # this is the target directory
            target_size=(320, 320),  # all images will be resized to 320*320
            batch_size=2,
            class_mode='categorical')


