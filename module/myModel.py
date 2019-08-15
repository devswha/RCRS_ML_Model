from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Reshape, Conv3D, Add, SeparableConv2D
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling3D
from keras.layers import LSTM, ConvLSTM2D, TimeDistributed, InputLayer, Lambda, BatchNormalization
from keras.models import Model
from keras.layers import merge, Input
from keras.regularizers import l2
from keras.activations import relu

from .resnet_v1 import resnet_v1_model
from .resnet_v2 import resnet_v2_model
from .inception_resnet_v2 import InceptionResNetV2_model
from .inception_v3 import InceptionV3_model
from .resnext import ResNext_model
from .mobilenets import MobileNet_model
from .resnet import ResNet_model
from .densenet import DenseNet_121_model
from .densenet import DenseNet_169_model
from .densenet import DenseNet_201_model
from .densenet import DenseNet_264_model
from .xception import Xception_model
from .wide_resnet import WideResNet_model
from keras import backend as K
import inspect
import io
import sys
import numpy as np
import tensorflow as tf


#### ResNet

def ResNet_50(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes)
    return model

def ResNet_50_CBAM(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='cbam_block')
    return model

def ResNet_50_SE(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='se_block')
    return model

def ResNet_50_GCBAM(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='gcbam_block')
    return model

def ResNet_50_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def ResNet_101(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 6, 23, 3], classes=num_classes)
    return model

def ResNet_101_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 6, 23, 3], classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNet_152(HEIGHT, WIDTH, num_classes):
    model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 8, 36, 3], classes=num_classes)
    return model

def ResNet_152_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 8, 36, 3], classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model




# ResNext

def ResNeXt_50(HEIGHT, WIDTH, num_classes):
    model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes)
    return model

def ResNeXt_50_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_50_CBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='cbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_50_SE_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='se_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_50_GCBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 6, 3], classes=num_classes, attention_module='gcbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_101(HEIGHT, WIDTH, num_classes):
    model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes)
    return model

def ResNeXt_101_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_101_CBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes, attention_module='cbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_101_SE_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes, attention_module='se_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def ResNeXt_101_GCBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = ResNext_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), depth=[3, 4, 23, 3], classes=num_classes, attention_module='gcbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model




def DenseNet_121(HEIGHT, WIDTH, num_classes):
    model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def DenseNet_121_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def DenseNet_121_CBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def DenseNet_121_SE_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def DenseNet_121_GCBAM_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_121_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def DenseNet_169(HEIGHT, WIDTH, num_classes):
    model = DenseNet_169_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def DenseNet_169_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_169_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def DenseNet_201(HEIGHT, WIDTH, num_classes):
    model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def DenseNet_201_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model

def DenseNet_201_CBAM(HEIGHT, WIDTH, num_classes):
    model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    return model

def DenseNet_201_SE(HEIGHT, WIDTH, num_classes):
    model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def DenseNet_201_GCBAM(HEIGHT, WIDTH, num_classes):
    model = DenseNet_201_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    return model

def DenseNet_264(HEIGHT, WIDTH, num_classes):
    model = DenseNet_264_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def DenseNet_264_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = DenseNet_264_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def InceptionResNet_v2(HEIGHT, WIDTH, num_classes):
    model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def InceptionResNet_v2_CBAM(HEIGHT, WIDTH, num_classes):
    model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    return model

def InceptionResNet_v2_SE(HEIGHT, WIDTH, num_classes):
    model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def InceptionResNet_v2_GCBAM(HEIGHT, WIDTH, num_classes):
    model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    return model

def InceptionResNet_v2_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = InceptionResNetV2_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def Inception_v3(HEIGHT, WIDTH, num_classes):
    model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def Inception_v3_CBAM(HEIGHT, WIDTH, num_classes):
    model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    return model

def Inception_v3_SE(HEIGHT, WIDTH, num_classes):
    model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def Inception_v3_GCBAM(HEIGHT, WIDTH, num_classes):
    model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    return model

def Inception_v3_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = InceptionV3_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model



def MobileNet(HEIGHT, WIDTH, num_classes):
    model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def MobileNet_CBAM(HEIGHT, WIDTH, num_classes):
    model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes,  attention_module='cbam_block')
    return model

def MobileNet_SE(HEIGHT, WIDTH, num_classes):
    model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def MobileNet_GCBAM(HEIGHT, WIDTH, num_classes):
    model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes,  attention_module='gcbam_block')
    return model

def MobileNet_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = MobileNet_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model


def Xception(HEIGHT, WIDTH, num_classes):
    model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes)
    return model

def Xception_CBAM(HEIGHT, WIDTH, num_classes):
    model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='cbam_block')
    return model

def Xception_SE(HEIGHT, WIDTH, num_classes):
    model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='se_block')
    return model

def Xception_GCBAM(HEIGHT, WIDTH, num_classes):
    model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module='gcbam_block')
    return model

def Xception_LSTM(HEIGHT, WIDTH, num_classes, seq_len):
    video = Input(shape=(seq_len, HEIGHT, WIDTH, 3), name='video_input')
    base_model = Xception_model(include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=num_classes, attention_module=None)
    base_model.trainable = False
    encoded_frame = TimeDistributed(base_model)(video)
    encoded_vid = LSTM(256)(encoded_frame)
    predictions = Dense(num_classes, activation='relu')(encoded_vid)
    model = Model(inputs=[video], outputs=predictions)
    return model
