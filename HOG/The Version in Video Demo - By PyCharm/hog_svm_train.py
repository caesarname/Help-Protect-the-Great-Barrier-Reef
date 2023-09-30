import glob
import random
import shutil
import joblib
from tqdm import tqdm
from sklearn.svm import LinearSVC
from os.path import exists
import os

trainNegFeaturesPath = "E:/COTS/train_images/train/feature64_64/neg/"
trainPosFeaturesPath = "E:/COTS/train_images/train/feature64_64/pos/"
trainHardFeaturePath = "E:/COTS/train_images/train/feature64_64/hard/"
#testPath = "E:/COTS/train_images/test"

negFeatList = glob.glob(trainNegFeaturesPath + '*.feature')
posFeatList = glob.glob(trainPosFeaturesPath + '*.feature')
hardFeatList = glob.glob(trainHardFeaturePath + '*.feature')

# https://blog.csdn.net/wavehaha/article/details/113484407
def clearDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)


#clearDirectory(testPath)

print(len(negFeatList))
# total 11898 bbox (features), choose 80% for train.
trainNegFeatList = random.sample(negFeatList, 90000)
# total 11898 bbox (features), choose 80% for train.
trainPosFeatList = random.sample(posFeatList, 9500)

# print(len(negFeatList))
# for image in tqdm(negFeatList):
#    #print(type(image))
#    if image not in trainNegFeatList:
#        #print("1")
#        sourceImg = "E:/COTS/train_images/train/neg/" + os.path.basename(image)[0:-10]
#        file_exists = exists("E:/COTS/train_images/test/" + os.path.basename(image)[0:-10])
#        if not file_exists:
#            #print("3")
#            sourceImg = sourceImg.strip("-")
#            shutil.copyfile(sourceImg, "E:/COTS/train_images/test/" +os.path.basename(image)[0:-10])

# for image in tqdm(posFeatList):
#    if image not in trainPosFeatList:
#        sourceImg = "E:/COTS/train_images/train/pos/" + os.path.basename(image)[0:-10]
#        file_exists = exists("E:/COTS/train_images/test/" + os.path.basename(image)[0:-10])
#        print(sourceImg)
#        print(file_exists)
#        if not file_exists:
#            #print("2")
#            sourceImg = sourceImg.strip("-")
#            shutil.copyfile(sourceImg, "E:/COTS/train_images/test/" +os.path.basename(image)[0:-10])

modelPath = "E:/COTS/train_images/train/svmWindow32_32_8822_90000Neg_9500Pos_unweight.model"
features = []
labels = []


# trying to capture positive features which has lower amounts
clf = LinearSVC(max_iter=100000)
print("load negFeatures")
for feat in tqdm(trainNegFeatList):
    feat_neg_data = joblib.load(feat)
    features.append(feat_neg_data)
    labels.append(0)

print("load posFeatures")
for feat in tqdm(trainPosFeatList):
    feat_pos_data = joblib.load(feat)
    features.append(feat_pos_data)
    labels.append(1)

#print("load hardFeatures")
#for feat in tqdm(trainHardFeatList):
#    feat_hard_data = joblib.load(feat)
#    features.append(feat_hard_data)
#    labels.append(0)
clf.fit(features, labels)
joblib.dump(clf, modelPath)
