import kagglehub
import cv2
import os
import numpy as np

def main() -> None:
    # Download dataset
    path = kagglehub.dataset_download("ahemateja19bec1025/traffic-sign-dataset-classification")

    # Load template image
    template = cv2.imread("templates/0.jpg")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(template, 200, 250)
    # edges = cv2.imread("edges.png")
    # template = edges #cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("edges.png", edges)

    ght = cv2.createGeneralizedHoughGuil()
    ght.setTemplate(template)

    ght.setMinDist(100)
    ght.setMinAngle(0)
    ght.setMaxAngle(360)
    ght.setAngleStep(90)
    ght.setLevels(360)
    ght.setMinScale(0.1)
    ght.setMaxScale(2)
    ght.setScaleStep(0.01)
    ght.setAngleThresh(1)
    ght.setScaleThresh(1)
    ght.setPosThresh(1)
    # ght.setAngleEpsilon(1)
    # ght.setXi(1)

    # Load image
    data_path = os.path.join(path, "traffic_Data", "DATA", "0")
    # img_path = os.path.join(data_path, "000_0056.png")
    img_path = "scene.jpg"
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Hough Transform
    positions = ght.detect(img_gray)
    print(positions)

    # Print positions
    height, width = template.shape[:2]

    for position in positions[0][0]:
        center_col = int(position[0])
        center_row = int(position[1])
        scale = position[2]
        angle = int(position[3])

        found_height = int(height * scale)
        found_width = int(width * scale)

        rectangle = ((center_col, center_row),
                     (found_width, found_height),
                     angle)

        box = cv2.boxPoints(rectangle)
        box = np.array(box).reshape((-1,1,2)).astype(np.int32)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 1)

        img[center_row, center_col] = 0, 0, 255

    cv2.imwrite("results.png", img)

if __name__ == "__main__":
    main()