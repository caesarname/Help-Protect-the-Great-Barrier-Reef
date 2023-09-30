import random
import joblib
import os
import glob
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import cv2
import pandas
import re
import copy
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


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


def sliding_window(image, window_size, step_size):
    for row in range(0, image.shape[0], step_size[0]):
        for col in range(0, image.shape[1], step_size[1]):
            yield (row, col, image[row:row + window_size[0], col:col + window_size[1]])


def nms(detections, threshold=.2):
    if len(detections) == 0:
        return []
    detections = sorted(detections, key=lambda detections: detections[2], reverse=True)
    new_detections = []
    new_detections.append(detections[0])
    del detections[0]
    for index, detection in enumerate(detections):
        flag = True
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > threshold:
                overlapping_small = False
                del detections[index]
                break
        if flag:
            new_detections.append(detection)
            del detections[index]
    return new_detections


def overlapping_area(detection_1, detection_2):
    try:
        x1_rect1 = detection_1[0]  # x top left
        x1_rect2 = detection_2[0]
        x2_rect1 = detection_1[0] + detection_1[2]  # x bottom right
        x2_rect2 = detection_2[0] + detection_2[2]
        y1_rect1 = detection_1[1]  # y top left
        y1_rect2 = detection_2[1]
        y2_rect1 = detection_1[1] + detection_1[3]  # y bottom right
        y2_rect2 = detection_2[1] + detection_2[3]
        x_overlap = max(0, min(x2_rect1, x2_rect2) - max(x1_rect1, x1_rect2))
        y_overlap = max(0, min(y2_rect1, y2_rect2) - max(y1_rect1, y1_rect2))
        overlap_area = x_overlap * y_overlap
        area_1 = detection_1[2] * detection_1[3]
        area_2 = detection_2[2] * detection_2[3]
        total_area = area_1 + area_2 - overlap_area
    except IndexError:
        print(detection_1)
        print(detection_2)
    return overlap_area / float(total_area)


bboxDic = getBoundingBoxFromCsv("E:/COTS/train.csv")
clf = joblib.load("E:/COTS/svmWindow32_32_8822_5000Hard_90000Neg_9500Pos_unweight.model")
test_image_path = "E:/COTS/train_images/test/"
testImList = glob.glob(test_image_path + '*.jpg')
print(len(testImList))
testImList = random.sample(testImList, 15)
print(len(testImList))
window_size = (32, 32)
step_size = (5, 5)

F2ForTestSet = 0
PrecisionForTestSet = 0
RecallForTestSet = 0
testImList = ["E:/COTS/train_images/train/pos/0-53.jpg"]

# https://github.com/SamPlvs/Object-detection-via-HOG-SVM/blob/master/testing_HOG_SVM.py
for test_image in tqdm(testImList):
    detections = []
    test_image_read = cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    test_image_read = cv2.resize(test_image_read, (640, 360))
    scale = 0
    downscale = 1.25
    for test_image_pyramid in pyramid_gaussian(test_image_read, downscale=downscale):
        if test_image_pyramid.shape[0] < window_size[0] or test_image_pyramid.shape[1] < window_size[1]:
            break
        for (row, col, sliding_image) in sliding_window(test_image_pyramid, window_size, step_size):
            if sliding_image.shape != window_size:
                continue
            sliding_image_hog = hog(sliding_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
            sliding_image_hog = sliding_image_hog.reshape(1, -1)
            pred = clf.predict(sliding_image_hog)
            if pred == 1:
                pred_prob = clf.decision_function(sliding_image_hog)
                (window_height, window_width) = window_size
                detections.append((int(col * downscale ** scale), int(row * downscale ** scale),
                                   int(window_width * downscale ** scale), int(window_height * downscale ** scale),
                                   pred_prob))
        scale += 1

    detections = sorted(detections, key=lambda x: x[2], reverse=True)
    detections_nms = nms(detections, threshold=0.2)

    # visualize---------------------------
    test_image_detect = cv2.imread(test_image)
    test_image_detect = cv2.resize(test_image_detect, (640, 360))
    for detect in detections_nms:
        x = detect[0]
        y = detect[1]
        width = detect[3]
        height = detect[4]
        print(x, y, width, height)
        cv2.rectangle(test_image_detect, pt1=(x, y), pt2=(x + width, y + height), color=(255, 0, 0), thickness=2)
    cv2.imshow(test_image, test_image_detect)
    cv2.waitKey(0)
    # end of visualize-----------------------

    # calculate F2
    # if nothing in detections
    if len(detections_nms) == 0:
        try:
            # if something in groundTruth
            bboxList = copy.deepcopy(bboxDic[os.path.basename(test_image)[0:-4]])
            # now we need to regard all box in groundTruth as FN
            F2Final = 0
            Recall = 0
            Precision = 0
            F2ForTestSet += F2Final
            print('nothing in detection but something in groundTruth')
        except KeyError:
            # nothing in predict and groundTruth
            F2Final = 1
            F2ForTestSet += F2Final
            PrecisionFinal = 1
            PrecisionForTestSet += PrecisionFinal
            RecallFinal = 1
            RecallForTestSet += RecallFinal
            print('nothing in detection and nothing in groundTruth')
    else:
        # if something in predict
        F2 = 0
        Precision = 0
        Recall = 0
        for iou in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            TP = 0
            FN = 0
            FP = 0
            try:
                # if something in groundTruth and predict
                bboxList = copy.deepcopy(bboxDic[os.path.basename(test_image)[0:-4]])
                realTrue = len(bboxList)
                matched = []
                for detection in detections_nms:
                    highestIOU = [0, ""]
                    # modify here in colab version, this is not the correct calculate method
                    #
                    for bbox in range(len(bboxList)):
                        IOU = overlapping_area(bboxList[bbox], detection)
                        print(IOU)
                        if IOU > highestIOU[0]:
                            highestIOU = [IOU, bboxList[bbox]]
                    if iou > highestIOU[0]:
                        TP += 1
                    else:
                        FP += 1
                for i in bboxList:
                    FN += 1
                F2 += 5 * TP / (5 * TP + 4 * FN + FP)
                Precision += TP / (TP + FP)
                Recall += TP / realTrue
            except KeyError:
                # Annotation has no bbox, but we detect it. marked as FP
                bboxList = []
                F2 = 0
                Precision = 0
                Recall = 0
                print('something in detection and nothing in groundTruth')
                break

        PrecisionFinal = Precision / 11
        RecallFinal = Recall / 11
        F2Final = F2 / 11
        F2ForTestSet += F2Final
        PrecisionForTestSet += PrecisionFinal
        RecallForTestSet += RecallFinal
        print(str(F2Final))
F2ForTestSet = F2ForTestSet / len(testImList)
PrecisionForTestSet = PrecisionForTestSet / len(testImList)
RecallForTestSet = RecallForTestSet / len(testImList)
print("F2 score is: " + str(F2ForTestSet))
# print("Precision score is: " + str(PrecisionForTestSet))
# print("Recall score is: " + str(RecallForTestSet))
