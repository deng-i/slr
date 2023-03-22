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
    GlobalAveragePooling2D,
    Add
)
from tensorflow.keras.models import Model
from train.Segmentation.Seg_Model import SegModel


class RecModel(object):
    def __init__(self, input_size, num_class):
        self.input_size = input_size
        self.num_class = num_class

        self._build_model()

    def _build_model(self):
        SegM = SegModel(self.input_size)
        Smodel = SegM.model
        Smodel.load_weights('train/Seg_weight.hdf5')
        layer_num = len(Smodel.layers)
        for layer in Smodel.layers[:layer_num]:
            layer.trainable = False

        inp = Input(shape=self.input_size)
        inp_stream1 = Smodel.input
        inp_stream2 = Smodel.output

        # appearance stream
        x = Conv2D(16, 3, activation='relu', padding='same', dilation_rate=1, name='CV1')(inp_stream1)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(32, 3, activation='relu', padding='same', dilation_rate=2, name='CV2')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2, name='CV3')(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)

        x = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=3, name='CV4')(x)
        x = GlobalAveragePooling2D()(x)

        x = Dropout(0.2)(x)
        x = Dense(64, activation="relu", name="Dense1")(x)
        #
        x = Dropout(0.3)(x)
        x = Dense(64, activation="relu", name="Dense2")(x)
        # xf1 = Dropout(0.75)(xf1)

        # shape stream
        x1 = Conv2D(16, 3, activation='relu', padding='same', dilation_rate=1, name='CV21')(inp_stream2)
        x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        x1 = Conv2D(32, 3, activation='relu', padding='same', dilation_rate=2, name='CV22')(x1)
        x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        x1 = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2, name='CV23')(x1)
        x1 = MaxPooling2D(pool_size=(3, 3))(x1)

        x1 = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=3, name='CV24')(x1)
        x1 = GlobalAveragePooling2D()(x1)

        x1 = Dropout(0.2)(x1)
        x1 = Dense(64, activation="relu", name="Dense21")(x1)
        #
        x1 = Dropout(0.3)(x1)
        x1 = Dense(64, activation="relu", name="Dense22")(x1)
        # xf2 = Dropout(0.45)(xf2)

        # Concatenate
        f = Add()([x, x1])
        # f = Dropout(0.45)(f)
        # f = Dense(64, activation="relu")(f)

        # Final Prediction
        prediction = Dense(self.num_class, activation='softmax')(f)
        model_final = Model(inputs=Smodel.input, outputs=prediction)

        self.model_F = model_final
