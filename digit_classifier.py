import kagglehub
import os
import numpy as np
import cv2
import argparse
import hough_circles
from preprocess import preprocess_image

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def main() -> None:
    # parser = argparse.ArgumentParser(description='Hough Transform')
    # parser.add_argument('-i', '--image', type=str, required=True, help='Image path')
    # args = parser.parse_args()

    data_path = kagglehub.dataset_download("ahemateja19bec1025/traffic-sign-dataset-classification")
    data_path = os.path.join(data_path, "traffic_Data", "DATA")
    
    classes = [str(x) for x in range(31)]
    # print(classes)

    X = []
    y = []

    for i in range(len(classes)):
        class_path = os.path.join(data_path, classes[i])
        for file in os.listdir(class_path):
            path = os.path.join(class_path, file)
            img = cv2.imread(path)
            X.append(preprocess_image(img))
            y.append(i)

    X = np.array(X)
    y = np.array(y)

    # print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    # knn.fit(X_train, y_train)

    # print(knn.score(X_test, y_test))
   
    total = 0
    correct = []
    for file in os.listdir("test"):
        _, detected_circles = hough_circles.get_circles(os.path.join("test", file))

        detected_circles = np.array([preprocess_image(img) for img in detected_circles])

        total += 1

        if len(detected_circles) == 0:
            print(file)
            continue

        y_pred = knn.predict(detected_circles)
        if int(file[:2]) == y_pred[-1]:
            correct.append(int(y_pred[-1]))
        else:
            print(file, y_pred)

        # if y_pred[-1] < 8:
        #     kph_pred = [class_to_kph[y] for y in y_pred]

        #     if int(file[:2]) == kph_pred[-1]:
        #         correct += 1

        #     print(file, kph_pred)
        # else:
        #     print(file, y_pred)

    correct.sort()
    print(correct)
    print(len(correct), total)


if __name__ == "__main__":
    main()

    