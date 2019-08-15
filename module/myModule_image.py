import os
import random
import shutil
import fileinput
import sys
import shutil
import subprocess
import signal
import time
import datetime

config = open("./config_image.txt", 'r')
lines = config.readlines()

for line in lines:
    ###########################################################
    # Label constant
    ###########################################################
    if "LIMIT_TIME_STEP" in line:
        LIMIT_TIME_STEP = int(line.split()[1])
    if "LIMIT_CIVILIAN_HP" in line:
        LIMIT_CIVILIAN_HP = int(line.split()[1])
    if "LABEL_TRAIN_START_MAP_NUM" in line:
        LABEL_TRAIN_START_MAP_NUM = int(line.split()[1])
    if "LABEL_TRAIN_END_MAP_NUM" in line:
        LABEL_TRAIN_END_MAP_NUM = int(line.split()[1])
    if "LABEL_TEST_START_MAP_NUM" in line:
        LABEL_TEST_START_MAP_NUM = int(line.split()[1])
    if "LABEL_TEST_END_MAP_NUM" in line:
        LABEL_TEST_END_MAP_NUM = int(line.split()[1])
    if "LABEL_DATASET_DIR" in line:
        LABEL_DATASET_DIR = os.path.abspath(line.split()[1])

    ###########################################################
    # Convert constant
    ###########################################################
    if "CONVERT_TRAIN_START_MAP_NUM" in line:
        CONVERT_TRAIN_START_MAP_NUM = int(line.split()[1])
    if "CONVERT_TRAIN_END_MAP_NUM" in line:
        CONVERT_TRAIN_END_MAP_NUM = int(line.split()[1])
    if "CONVERT_TEST_START_MAP_NUM" in line:
        CONVERT_TEST_START_MAP_NUM = int(line.split()[1])
    if "CONVERT_TEST_END_MAP_NUM" in line:
        CONVERT_TEST_END_MAP_NUM = int(line.split()[1])
    if "CONVERT_DATASET_DIR" in line:
        CONVERT_DATASET_DIR = os.path.abspath(line.split()[1])

    ###########################################################
    # ML constant
    ###########################################################
    if "MODEL_TRAIN_NAME" in line:
        MODEL_TRAIN_NAME = line.split()[1]
    if "MODEL_TRAIN_START_MAP_NUM" in line:
        MODEL_TRAIN_START_MAP_NUM = int(line.split()[1])
    if "MODEL_TRAIN_END_MAP_NUM" in line:
        MODEL_TRAIN_END_MAP_NUM = int(line.split()[1])
    if "MODEL_TEST_NAME" in line:
        MODEL_TEST_NAME = line.split()[1]
    if "MODEL_TEST_START_MAP_NUM" in line:
        MODEL_TEST_START_MAP_NUM = int(line.split()[1])
    if "MODEL_TEST_END_MAP_NUM" in line:
        MODEL_TEST_END_MAP_NUM = int(line.split()[1])
    if "MODELS_DIR" in line:
        MODELS_DIR = os.path.abspath(line.split()[1])
    if "MODEL_DATASET_DIR" in line:
        MODEL_DATASET_DIR = os.path.abspath(line.split()[1])
    if "RESULTS_DIR" in line:
        RESULTS_DIR = os.path.abspath(line.split()[1])

SIMULATOR_DIR = os.path.abspath('./RCRS/simulator')
BASE_MAP_DIR = os.path.abspath('./RCRS/baseMap')

config.close()
