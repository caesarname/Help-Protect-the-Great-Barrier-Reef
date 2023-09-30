import random

import joblib
import os
import glob
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import cv2
import shutil
import copy
from tqdm import tqdm


def sliding_window(image, window_size, step_size):
    for row in range(0, image.shape[0], step_size[0]):
        for col in range(0, image.shape[1], step_size[1]):
            yield (row, col, image[row:row + window_size[0], col:col + window_size[1]])


# https://blog.csdn.net/wavehaha/article/details/113484407
def clearDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


clf = joblib.load("E:/COTS/svmWindow32_32_8822_9500Hard_80000neg_9500Pos_unbalanced.model")

trainHardImagePath = "E:/COTS/train_images/train/neg/"
trainHardPath = "E:/COTS/train_images/train/feature64_64/hard/"

negImgList = glob.glob(trainHardImagePath + '*.jpg')

negImgList = random.sample(negImgList, 8000)
window_size = (32, 32)
step_size = (5, 5)

clearDirectory(trainHardPath)
for image in tqdm(negImgList):
    test_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (640, 360))
    scale = 0
    detections = []
    downscale = 1.25
    index = 0
    for test_image_pyramid in pyramid_gaussian(test_image, downscale=downscale):
        if test_image_pyramid.shape[0] < window_size[0] or test_image_pyramid.shape[1] < window_size[1]:
            break
        for (row, col, sliding_image) in sliding_window(test_image_pyramid, window_size, step_size):
            if sliding_image.shape != window_size:
                continue
            if index == 10:
                break
            sliding_image_hog = hog(sliding_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
            copyOfHog = copy.deepcopy(sliding_image_hog)
            sliding_image_hog = sliding_image_hog.reshape(1, -1)
            pred = clf.predict(sliding_image_hog)
            if pred == 1:
                index += 1
                featName = os.path.basename(image) + "-" + "hard" + "-" + str(index) + '.feature'
                joblib.dump(copyOfHog, trainHardPath + "/" + featName)

        scale += 1
    if index == 0:
        print("detect nothing in "+str(os.path.basename(image)))
