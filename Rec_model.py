from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Convolution2D,
    MaxPooling2D,
    BatchNormalization,
    Conv2D,
    Input,
    AveragePooling2D,
    concatenate,
)
from tensorflow.keras.models import Model
from Segmentation.Seg_Model import SegModel


class RecModel(object):
    def __init__(self, input_size, num_class):
        self.input_size = input_size
        self.num_class = num_class

        self._build_model()

    def _build_model(self):
        SegM = SegModel(self.input_size)
        Smodel = SegM.model
        Smodel.load_weights('Seg_weight.hdf5')
        layer_num = len(Smodel.layers)
        for layer in Smodel.layers[:layer_num]:
            layer.trainable = False

        inp = Input(shape=self.input_size)
        inp_stream1 = Smodel.input
        inp_stream2 = Smodel.output

        # Stream1
        x = Conv2D(16, 3, activation='relu', padding='same', dilation_rate=1, name='CV1')(inp_stream1)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(32, 3, activation='relu', padding='same', dilation_rate=2, name='CV2')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2, name='CV4')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=3, name='CV41')(x)
        xf1 = MaxPooling2D(pool_size=(3, 3))(x)

        # Stream2
        x1 = Conv2D(16, 3, activation='relu', padding='same', dilation_rate=1, name='CV11')(inp_stream2)
        x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        x1 = Conv2D(32, 3, activation='relu', padding='same', dilation_rate=2, name='CV421')(x1)
        x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        x1 = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2, name='CV31')(x1)
        x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        x1 = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=3, name='CV412')(x1)
        xf2 = MaxPooling2D(pool_size=(3, 3))(x1)

        # Concatenate and Pooling
        f = concatenate([xf1, xf2], axis=3)  # !!!!!!!!!!!!!!!!!
        f = MaxPooling2D(pool_size=(3, 3))(f)

        # Flattening and Dropout
        f = Flatten()(f)
        f = Dropout(0.2)(f)

        # Final Prediction
        prediction = Dense(self.num_class, activation='softmax')(f)
        model_final = Model(inputs=Smodel.input, outputs=prediction)

        self.model_F = model_final
