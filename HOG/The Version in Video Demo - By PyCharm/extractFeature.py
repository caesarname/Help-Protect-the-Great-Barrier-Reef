import os
import cv2
import numpy as np
import shutil
from skimage.feature import hog
import glob
import joblib
import pandas
import re
from tqdm import tqdm


# importrant note1: please use x64 version instead x32 version python when running this code
# x32 will lead limit memory usage for extraction of image hog feature64_64
# importrant note2: try to allocate as much as possible RAM (or virtual memory) to pycharm and it`s location disk

trainNegPath = "E:/COTS/train_images/train/neg/"
trainPosPath = "E:/COTS/train_images/train/pos/"

trainNegFeaturesPath = "E:/COTS/train_images/train/feature64_64/neg"
trainPosFeaturesPath = "E:/COTS/train_images/train/feature64_64/pos"


# https://blog.csdn.net/wavehaha/article/details/113484407
def clearDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


def getBoundingBoxFromCsv(path):
    bboxfile = pandas.read_csv(path)
    bboxDic = {}
    for j in range(len(bboxfile)):
        if bboxfile["annotations"][j] == "[]":
            pass
        else:
            bboxOut = []
            bbox = bboxfile.iloc[j, [5]].tolist()
            temp = re.split("[\\s*$&#/\"'\\,.:;?!\\[\\]{}()<>~\\-_]+", bbox[0])
            temp = [i for i in temp if i != '']
            while len(temp) != 0:
                x = temp[1]
                y = temp[3]
                width = temp[5]
                height = temp[7]
                bboxOut.append((x, y, width, height))
                del temp[0:8]
            image_id = bboxfile.iloc[j, [4]].to_string().split(' ')
            image_id = image_id[-1]
            bboxDic[image_id] = bboxOut
    return bboxDic


clearDirectory(trainNegFeaturesPath)
clearDirectory(trainPosFeaturesPath)
bboxDic = getBoundingBoxFromCsv("E:/COTS/train.csv")
trainNegImg = glob.glob(trainNegPath + '*.jpg')
trainPosImg = glob.glob(trainPosPath + '*.jpg')


posF = []
negF = []

print("Do PosImg feature64_64 extraction, and with image resize to 640*360(half)")

for posImg in tqdm(trainPosImg):
    posImRead = cv2.imread(posImg)
    posImRead = cv2.resize(posImRead, (640, 360))
    bboxList = bboxDic[posImg.split("\\")[1][0:-4]]
    for index in range(len(bboxList)):

        y = round(int(bboxList[index][1])/2)
        x = round(int(bboxList[index][0])/2)
        height = round(int(bboxList[index][3])/2)
        width = round(int(bboxList[index][2])/2)
        bboxImg = posImRead[y:y + height, x:x + width]
        bboxImg = cv2.resize(bboxImg, (32, 32))
        feat = hog(bboxImg, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
        featName = os.path.basename(posImg) + "-" + str(index) + '.feature'
        joblib.dump(feat, trainPosFeaturesPath + "/" + featName)

# https://stackoverflow.com/questions/42263020/opencv-trying-to-get-random-portion-of-image
print("Do NegImg feature64_64 extraction")
for negImg in tqdm(trainNegImg):
    temp = cv2.imread(negImg)
    # random crop 1 image in one neg image.
    for i in range(0, 5):
        max_x = temp.shape[1] - 32
        max_y = temp.shape[0] - 32
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        negImRead = temp[y:y + 32,x:x + 32]
        # cv2.imshow("?", negImRead)
        # cv2.waitKey(0)
        feat = hog(negImRead, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
        featName = os.path.basename(negImg) + "_" + str(i) + '.feature'
        joblib.dump(feat, trainNegFeaturesPath + "/" + featName)
