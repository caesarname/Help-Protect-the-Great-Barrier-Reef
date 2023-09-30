import pandas as pd
import os
import shutil


# https://blog.csdn.net/wavehaha/article/details/113484407
def clearDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)


# 1.Read train file. split it into train and test files.
trainExcel = pd.read_csv("train.csv")

negImgName = []
posImgName = []

# get all the negative and positive image id
for j in range(len(trainExcel)):
    if trainExcel["annotations"][j] == "[]":
        temp = trainExcel.iloc[j, [4]].to_string().split(' ')
        image_id = temp[-1]
        negImgName.append(image_id)

    else:
        temp = trainExcel.iloc[j, [4]].to_string().split(' ')
        image_id = temp[-1]
        # print(image_id)
        posImgName.append(image_id)

    # choose only video0 for first time train and test
negImgNameVideo0 = []
negImgNameVideo1 = []
negImgNameVideo2 = []
posImgNameVideo0 = []
posImgNameVideo1 = []
posImgNameVideo2 = []

# convenient for split image
for name in negImgName:
    if name[0] == "0":
        negImgNameVideo0.append(name[2:])
    elif name[0] == "1":
        negImgNameVideo1.append(name[2:])
    elif name[0] == "2":
        negImgNameVideo2.append(name[2:])

for name in posImgName:
    if name[0] == "0":
        posImgNameVideo0.append(name[2:])
    elif name[0] == "1":
        posImgNameVideo1.append(name[2:])
    elif name[0] == "2":
        posImgNameVideo2.append(name[2:])

trainNegTarget = "E:/COTS/train_images/train/neg"
trainPosTarget = "E:/COTS/train_images/train/pos"

clearDirectory(trainNegTarget)
clearDirectory(trainPosTarget)

# do the transfer
for negImage in negImgNameVideo0:
    sourceImg = "E:/COTS/train_images/video_0/" + str(negImage) + ".jpg"
    shutil.copyfile(sourceImg, trainNegTarget + "/" + "0-" + str(negImage) + ".jpg")
for posImage in posImgNameVideo0:
    sourceImg = "E:/COTS/train_images/video_0/" + str(posImage) + ".jpg"
    shutil.copyfile(sourceImg, trainPosTarget + "/" + "0-" + str(posImage) + ".jpg")
for negImage in negImgNameVideo1:
    sourceImg = "E:/COTS/train_images/video_1/" + str(negImage) + ".jpg"
    shutil.copyfile(sourceImg, trainNegTarget + "/" + "1-" + str(negImage) + ".jpg")
for posImage in posImgNameVideo1:
    sourceImg = "E:/COTS/train_images/video_1/" + str(posImage) + ".jpg"
    shutil.copyfile(sourceImg, trainPosTarget + "/" + "1-" + str(posImage) + ".jpg")
for negImage in negImgNameVideo2:
    sourceImg = "E:/COTS/train_images/video_2/" + str(negImage) + ".jpg"
    shutil.copyfile(sourceImg, trainNegTarget + "/" + "2-" + str(negImage) + ".jpg")
for posImage in posImgNameVideo2:
    sourceImg = "E:/COTS/train_images/video_2/" + str(posImage) + ".jpg"
    shutil.copyfile(sourceImg, trainPosTarget + "/" + "2-" + str(posImage) + ".jpg")
