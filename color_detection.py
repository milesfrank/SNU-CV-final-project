import cv2
import numpy as np
import os
import kagglehub
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

path = kagglehub.dataset_download("ahemateja19bec1025/traffic-sign-dataset-classification")
def createColorHistogram(image):
    # define colors to plot the histograms 
    colors = ('b','g','r') 
    
    # compute and plot the image histograms 
    for i,color in enumerate(colors): 
        hist = cv2.calcHist([image],[i],None,[256],[0,256]) 
        #plt.plot(hist,color = color) 
    # plt.title('Image Histogram') 
    # plt.show()

    return hist.flatten()

def knnTrainCircleSigns():
    features = []
    labels = []

    # Speed Limit
    for i in tqdm(range(8)):
        dir = path + "/traffic_Data/DATA/" + str(i)
        for file in os.listdir(dir):
            img = cv2.imread(dir + "/" + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hist = createColorHistogram(img)

            features.append(hist)
            labels.append("SpeedLimit")

    # Turning Signs 
    for i in tqdm(range(8, 14)):
        dir = path + "/traffic_Data/DATA/" + str(i)
        for file in os.listdir(dir):
            img = cv2.imread(dir + "/" + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hist = createColorHistogram(img)

            features.append(hist)
            labels.append("Turning")

    # Blue Turning Signs 
    for i in tqdm(range(20, 27)):
        dir = path + "/traffic_Data/DATA/" + str(i)
        for file in os.listdir(dir):
            img = cv2.imread(dir + "/" + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hist = createColorHistogram(img)

            features.append(hist)
            labels.append("BlueTurning")

    # Blue X Signs 
    dir = path + "/traffic_Data/DATA/" + str(54)
    for file in os.listdir(dir):
        img = cv2.imread(dir + "/" + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = createColorHistogram(img)

        features.append(hist)
        labels.append("BlueX")

    # Warning
    dir = path + "/traffic_Data/DATA/" + str(55)
    for file in os.listdir(dir):
        img = cv2.imread(dir + "/" + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = createColorHistogram(img)

        features.append(hist)
        labels.append("Warning")

    features = np.array(features)
    labels = np.array(labels)

    # Create train/test split
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

    # Train and test model
    model = KNeighborsClassifier()
    model.fit(trainFeat, trainLabels)
    acc = model.score(testFeat, testLabels)
    print(acc)

    return model

def knnTrainTriangleSigns():
    features = []
    labels = []

    # Traffic Light
    dir = path + "/traffic_Data/DATA/" + str(33)
    for file in os.listdir(dir):
        img = cv2.imread(dir + "/" + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = createColorHistogram(img)

        features.append(hist)
        labels.append("TrafficLight")

    # Warning
    for i in tqdm(range(34, 47)):
        dir = path + "/traffic_Data/DATA/" + str(i)
        for file in os.listdir(dir):
            img = cv2.imread(dir + "/" + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hist = createColorHistogram(img)

            features.append(hist)
            labels.append("Warning")

    # Yield
    dir = path + "/traffic_Data/DATA/" + str(56)
    for file in os.listdir(dir):
        img = cv2.imread(dir + "/" + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = createColorHistogram(img)

        features.append(hist)
        labels.append("Yield")

    # Traffic Light
    dir = path + "/traffic_Data/DATA/" + str(33)
    for file in os.listdir(dir):
        img = cv2.imread(dir + "/" + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = createColorHistogram(img)

        features.append(hist)
        labels.append("TrafficLight")

    features = np.array(features)
    labels = np.array(labels)

    # Create train/test split
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

    # Train and test model
    model = KNeighborsClassifier()
    model.fit(trainFeat, trainLabels)
    acc = model.score(testFeat, testLabels)
    print(acc)

    return model

if __name__ == "__main__":
    circleClassifier = knnTrainCircleSigns()
    triangleClassifier = knnTrainTriangleSigns()

    # image = path + "/traffic_Data/DATA/46/046_1_0001.png"
    # img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hist = createColorHistogram(img)
    # print(triangleClassifier.predict([hist]))
