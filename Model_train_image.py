from __future__ import absolute_import, division, print_function
from module.myModule_image import *
from module.myModel import *
from tensorflow import reset_default_graph
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import keras
import inspect
from sklearn.model_selection import KFold


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def callModel(model_name, HEIGHT, WIDTH, num_classes):
    if model_name == 'ResNet_50':
        return ResNet_50(HEIGHT, WIDTH, num_classes)
    if model_name == 'ResNet_50_CBAM':
        return ResNet_50_CBAM(HEIGHT, WIDTH, num_classes)
    if model_name == 'ResNet_50_SE':
        return ResNet_50_SE(HEIGHT, WIDTH, num_classes)
    if model_name == 'ResNet_50_GCBAM':
        return ResNet_50_GCBAM(HEIGHT, WIDTH, num_classes)
    if model_name == 'ResNet_101':
        return ResNet_101(HEIGHT, WIDTH, num_classes)
    if model_name == 'ResNet_152':
        return ResNet_152(HEIGHT, WIDTH, num_classes)

    if model_name == 'ResNeXt_50':
        return ResNeXt_50(HEIGHT, WIDTH, num_classes)
    if model_name == 'ResNeXt_101':
        return ResNeXt_101(HEIGHT, WIDTH, num_classes)

    if model_name == 'DenseNet_121':
        return DenseNet_121(HEIGHT, WIDTH, num_classes)
    if model_name == 'DenseNet_169':
        return DenseNet_169(HEIGHT, WIDTH, num_classes)
    if model_name == 'DenseNet_201':
        return DenseNet_201(HEIGHT, WIDTH, num_classes)
    if model_name == 'DenseNet_201_CBAM':
        return DenseNet_201_CBAM(HEIGHT, WIDTH, num_classes)
    if model_name == 'DenseNet_201_SE':
        return DenseNet_201_SE(HEIGHT, WIDTH, num_classes)
    if model_name == 'DenseNet_201_GCBAM':
        return DenseNet_201_GCBAM(HEIGHT, WIDTH, num_classes)
    if model_name == 'DenseNet_264':
        return DenseNet_264(HEIGHT, WIDTH, num_classes)

    if model_name == 'InceptionResNet_v2':
        return InceptionResNet_v2(HEIGHT, WIDTH, num_classes)
    if model_name == 'InceptionResNet_v2_CBAM':
        return InceptionResNet_v2_CBAM(HEIGHT, WIDTH, num_classes)
    if model_name == 'InceptionResNet_v2_SE':
        return InceptionResNet_v2_SE(HEIGHT, WIDTH, num_classes)
    if model_name == 'InceptionResNet_v2_GCBAM':
        return InceptionResNet_v2_GCBAM(HEIGHT, WIDTH, num_classes)

    if model_name == 'Inception_v3':
        return Inception_v3(HEIGHT, WIDTH, num_classes)
    if model_name == 'Inception_v3_CBAM':
        return Inception_v3_CBAM(HEIGHT, WIDTH, num_classes)
    if model_name == 'Inception_v3_SE':
        return Inception_v3_SE(HEIGHT, WIDTH, num_classes)
    if model_name == 'InceptionResNet_v2_SE':
        return InceptionResNet_v2_SE(HEIGHT, WIDTH, num_classes)
    if model_name == 'Inception_v3_GCBAM':
        return Inception_v3_GCBAM(HEIGHT, WIDTH, num_classes)

    if model_name == 'MobileNet':
        return MobileNet(HEIGHT, WIDTH, num_classes)
    if model_name == 'MobileNet_CBAM':
        return MobileNet_CBAM(HEIGHT, WIDTH, num_classes)
    if model_name == 'MobileNet_SE':
        return MobileNet_SE(HEIGHT, WIDTH, num_classes)
    if model_name == 'MobileNet_GCBAM':
        return MobileNet_GCBAM(HEIGHT, WIDTH, num_classes)

    if model_name == 'Xception':
        return Xception(HEIGHT, WIDTH, num_classes)
    if model_name == 'Xception_CBAM':
        return Xception_CBAM(HEIGHT, WIDTH, num_classes)

    else:
        print("Error: you input the wrong model name !")
        exit(1)


mapName = ""
GRID_ROW = ""
GRID_COL = ""

if not len(sys.argv) is 4:
    print("Usage : python Model_train_image.py [Map name] [Grid row] [Grid col]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])

# Load path/class_id image file:
grid = "%dx%d" % (GRID_ROW, GRID_COL)
npyDir = "%s/image/train/%s/%s" % (MODEL_TRAIN_DATASET_DIR, mapName, grid)
if not os.path.exists("%s/image/%s" % (MODELS_DIR, mapName)):
    os.makedirs("%s/image/%s" % (MODELS_DIR, mapName))

#####################################################
# Train DNN model
#####################################################

batch_size = 8
nb_epoch = 15
num_classes = GRID_ROW * GRID_COL
WIDTH = 256
HEIGHT = 192

# Load train data
trainImage = []
for dataSetNum in range(1, int(MODEL_TRAIN_END_MAP_NUM/100)+1):
    trainImageData = np.load('%s/train_data_image-%d.npy' % (npyDir, dataSetNum))
    trainImage.extend(trainImageData[:])
train_X = np.array(trainImage)

trainLabel = []
for dataSetNum in range(1, int(MODEL_TRAIN_END_MAP_NUM/100)+1):
    trainLabelData = np.load('%s/train_data_label-%d.npy' % (npyDir, dataSetNum))
    trainLabel.extend(trainLabelData[:])
train_Y = np.array(trainLabel)

print("train_X shape: ", train_X.shape)
print("train_Y shape: ", train_Y.shape)


modelName = MODEL_TRAIN_NAME
model = callModel(modelName, HEIGHT, WIDTH, num_classes)
print(modelName)

save_dir = os.path.join(os.getcwd(), '%s/image/%s/%s_saved_models' % (MODELS_DIR, mapName, grid))

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, modelName)

model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='mean_squared_error', metrics=['mse'])
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0,
                                               mode='auto', baseline=None, restore_best_weights=True)
history = model.fit(train_X, train_Y, validation_split=0.2,
                     shuffle=True, callbacks=callbacks, epochs=nb_epoch, batch_size=batch_size)
model.save("%s/image/%s/%s_%s.h5" % (MODELS_DIR, mapName, grid, modelName))
