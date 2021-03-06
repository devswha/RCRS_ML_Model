from module.myModule_video import *
from sklearn.metrics import mean_squared_error
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
from math import sqrt
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.optimizers import Adam

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def lr_schedule(epoch):
    lr = 1e-3
    if epoch >= 15:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr


mapName = ""
GRID_ROW = ""
GRID_COL = ""

# Check data directory
if not len(sys.argv) is 5:
    print("Usage : python test_model.py [MapName] [GridRow] [GridCol] [predict_frame]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])
    predict_frame = int(sys.argv[4])

grid = "%dx%d" % (GRID_ROW, GRID_COL)
npyDir = "%s/video/test/%s/%s_%d" % (MODEL_TEST_DATASET_DIR, mapName, grid, predict_frame)
if not os.path.exists("%s/video/%s" % (RESULTS_DIR, mapName)):
    os.makedirs("%s/video/%s" % (RESULTS_DIR, mapName))

# Generator
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=8, dim=(192,256,3), n_channels=3,
                 n_classes=16, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=object)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(ID + '.npy')
            # Store class
            y[i] = self.labels[ID]

        return X, y

def callModel(model_name, HEIGHT, WIDTH, num_classes, seq):
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
    if model_name == 'ResNeXt_101_LSTM':
        return ResNeXt_101_LSTM(HEIGHT, WIDTH, num_classes, seq)

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

#####################################################
# Test DNN model
#####################################################

WIDTH = 256
HEIGHT = 192
num_classes = GRID_COL * GRID_ROW
seq = 9

# Load test data
testImage = []
partition = {}
partition['test'] = []
labels = {}

for dataSetNum in range(MODEL_TEST_START_MAP_NUM, MODEL_TEST_END_MAP_NUM+1):
    partition['test'].append('%s/test_data_video-%d' % (npyDir, dataSetNum))

testLabel = []
for dataSetNum in range(MODEL_TEST_START_MAP_NUM, MODEL_TEST_END_MAP_NUM+1):
    testLabelData = np.load('%s/test_data_label-%d.npy' % (npyDir, dataSetNum))
    labels['%s/test_data_video-%d' % (npyDir, dataSetNum)] = np.array(testLabelData, dtype=object)
    testLabel.extend(testLabelData[:])
testLabel = np.array(testLabel)


# Parameters
batch_size = 16
params = {'dim': (seq, HEIGHT, WIDTH),
          'batch_size': batch_size,
          'n_classes': num_classes,
          'n_channels': 3,
          'shuffle': True}

# Generators
testing_generator = DataGenerator(partition['test'], labels, **params)


# Test

modelName = MODEL_TEST_NAME


with CustomObjectScope({'relu6': keras.layers.advanced_activations.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = keras.models.load_model("%s/video/%s_%d_saved_models/%s" % (MODELS_DIR, grid, predict_frame, modelName), compile=False)
model.summary()
predictions = model.predict_generator(testing_generator, steps=500, workers=6)


index = 0
totalRMS = 0
rmsList = []
f = open("%s/video/%s/%s_%d_%s_RMSE.txt" % (RESULTS_DIR, mapName, grid, predict_frame, modelName), 'w')
for testSetNum in range(MODEL_TEST_START_MAP_NUM-1, MODEL_TEST_END_MAP_NUM):
    rms = sqrt(mean_squared_error(predictions[testSetNum], testLabel[testSetNum]))
    f.write(str(rms) + '\n')
    rmsList.append(rms)
    totalRMS = totalRMS + rms
    print(testSetNum)

totalRMS = totalRMS / MODEL_TEST_END_MAP_NUM
maxRMSList = []
for i in range(0,5):
    maxRMS = max(rmsList)
    maxRMSList.append(maxRMS)
    rmsList.remove(maxRMS)

avgMaxRMS = 0
for i in maxRMSList:
    avgMaxRMS = avgMaxRMS + i

print("ModelName: " + modelName)
print("NPY directory: " + npyDir)
print("top5RMSE: " + str(avgMaxRMS/5))
print("avgRMSE: " + str(totalRMS))
f.close()
