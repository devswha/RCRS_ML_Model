from module.myModule_video import *
import numpy as np
import cv2

mapName = ""
GRID_ROW = ""
GRID_COL = ""

# How long will you predict the next frame?
predict_frame = 1

if not len(sys.argv) is 5:
    print("Usage : python Converter_train_video.py [Map name] [Grid row] [Grid col] [predict_frame]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])
    predict_frame = int(sys.argv[4])

# Load path/class_id video file:
grid = "%dx%d" % (GRID_ROW, GRID_COL)
npyDir = "%s/video/train/%s/%s_%s" % (CONVERT_TRAIN_DATASET_DIR, mapName, grid, predict_frame)
if not os.path.exists(npyDir):
    os.makedirs(npyDir)


#######################################
# Create numpy training data
#######################################
WIDTH = 256
HEIGHT = 192
seq_len = int(SEQUENCE_TIME_STEP) # The length of sequence images to train
pathAndLabelListFile = open("%s/pathAndLabelListFile.txt" % npyDir, "w+")
npyTrainImageData = []
npyTrainLabelData = []
imageNumIndex = 0
npyNumIndex = 1

print("NPYDIR: ", npyDir)
print("SEQ_LEN: ", seq_len)

for dataSetNum in range(CONVERT_TRAIN_START_MAP_NUM, CONVERT_TRAIN_END_MAP_NUM + 1):
    # Select random time step
    pathList = []
    labelList = []
    dataSetPath = "%s/raw/train/generated_image/%s/%s_%d" % (CONVERT_TRAIN_DATASET_DIR, mapName, mapName, dataSetNum)
    rawImgListFile = open("%s/Label/%s/ImageList.txt" % (dataSetPath, grid), "r")
    for line in rawImgListFile.readlines():
        pathList.append("%s/Image/%s" % (dataSetPath, line.split(' ')[0].split('/')[-1]))
        label = line.split(' ')[1:]
        label[-1] = label[-1].rstrip('\n')
        label = list(map(int, label))
        labelList.append(label)

    for i in range(seq_len, 160, 10):
        # Save video to npy
        npyTrainSequenceData = []
        for index in range(i-seq_len, i):
            try:
                screen = cv2.imread(pathList[index], cv2.IMREAD_COLOR)
                screen = cv2.resize(screen, (WIDTH, HEIGHT))
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                npyTrainSequenceData.append(screen)
            except Exception as e:
                print(str(e))
        npyTrainImageData.append(npyTrainSequenceData)

        # Save label to npy
        npyTrainLabelData.append(labelList[i+predict_frame-1])

        # video
        trainImageDataFile = "%s/train_data_video-%d.npy" % (npyDir, npyNumIndex)
        np.save(trainImageDataFile, npyTrainImageData)
        print(trainImageDataFile)
        npyTrainImageData = []

        # label
        trainLabelDataFile = "%s/train_data_label-%d.npy" % (npyDir, npyNumIndex)
        np.save(trainLabelDataFile, npyTrainLabelData)
        print(trainLabelDataFile)
        npyTrainLabelData = []

        npyNumIndex = npyNumIndex + 1
