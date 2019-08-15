from module.myModule_image import *
from module.myModel import *
from sklearn.metrics import mean_squared_error
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
from math import sqrt
import numpy as np
import cv2
import tensorflow as tf
import keras

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

mapName = ""
GRID_ROW = ""
GRID_COL = ""

# Check data directory
if not len(sys.argv) is 4:
    print("Usage : python Model_test_image.py [MapName] [GridRow] [GridCol]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
grid = "%dx%d" % (GRID_ROW, GRID_COL)
npyDir = "%s/image/test/%s/%s" % (MODEL_DATASET_DIR, mapName, grid)

if not os.path.exists("%s/image/%s" % (RESULTS_DIR, mapName)):
    os.makedirs("%s/image/%s" % (RESULTS_DIR, mapName))

#####################################################
# Test DNN model
#####################################################

# Load test data
testImage = []
for dataSetNum in range(1, int(MODEL_TEST_END_MAP_NUM/100)+1):
    testImageData = np.load('%s/test_data_image-%d.npy' % (npyDir, dataSetNum))
    testImage.extend(testImageData[:])
test_X = np.array(testImage)

testLabel = []
for dataSetNum in range(1, int(MODEL_TEST_END_MAP_NUM/100)+1):
    testLabelData = np.load('%s/test_data_label-%d.npy' % (npyDir, dataSetNum))
    testLabel.extend(testLabelData[:])
test_Y = np.array(testLabel)

print("test_X shape: ", test_X.shape)
print("test_Y shape: ", test_Y.shape)

# Train the model
WIDTH = 256
HEIGHT = 192
num_classes = GRID_COL * GRID_ROW
modelName = MODEL_TEST_END_MAP_NUM
with CustomObjectScope({'relu6': keras.layers.advanced_activations.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = keras.models.load_model("%s/image/%s/%s/%s.h5" % (MODELS_DIR, mapName, grid, modelName), compile=False)

# Print model's summary and prediction accuracy
predictions = model.predict(test_X)
model.summary()

index = 0
totalRMS = 0
rmsList = []
f = open("%s/image/%s/%s_%s_RMSE.txt" % (RESULTS_DIR, mapName, grid, modelName), 'w')
for testSetNum in range(MODEL_TEST_START_MAP_NUM, MODEL_TEST_END_MAP_NUM + 1):
    rms = sqrt(mean_squared_error(predictions[testSetNum-1], test_Y[testSetNum-1]))
    f.write(str(rms) + '\n')
    rmsList.append(rms)
    totalRMS = totalRMS + rms

totalRMS = totalRMS / MODEL_TEST_END_MAP_NUM
maxRMSList = []
for i in range(0,5):
    maxRMS = max(rmsList)
    maxRMSList.append(maxRMS)
    rmsList.remove(maxRMS)

avgMaxRMS = 0
for i in maxRMSList:
    avgMaxRMS = avgMaxRMS + i

print("Model Name: " + modelName)
print("NPY Directory: " + npyDir)
print("top5RMSE: " + str(avgMaxRMS/5))
print("avgRMSE: " + str(totalRMS))
f.close()
