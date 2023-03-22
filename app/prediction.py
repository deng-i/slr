import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/trained_model", compile=False)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")

image = tf.keras.preprocessing.image.load_img(r"C:\Users\hogyv\OneDrive\Képek\Filmtekercs\asd.jpg")
image = tf.keras.preprocessing.image.img_to_array(image)
print(image.shape)
image = image * 1. / 255
print(image.shape)
# print(tf.reduce_min(image), tf.reduce_max(image))
print(model(np.expand_dims(image, axis=0)))
