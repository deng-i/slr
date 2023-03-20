from train.Rec_model import RecModel
import tensorflow as tf


class Prediction:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        for layer in self.model.layers[:-1]:
            layer.trainable = False
        prediction = tf.keras.layers.Dense(4, activation="softmax")(self.model.layers[-2].output)
        self.transfer_model = tf.keras.models.Model(inputs=self.model.input, outputs=prediction)

