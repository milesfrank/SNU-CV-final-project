import kagglehub
import cv2

def main() -> None:
    # Download dataset
    path = kagglehub.dataset_download("ahemateja19bec1025/traffic-sign-dataset-classification")

    # Load template image
    template = cv2.imread("templates/0.png")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    ght = cv2.createGeneralizedHoughGuil()
    ght.setTemplate(template)

    ght.setMinDist(100)
    ght.setMinAngle(0)
    ght.setMaxAngle(360)
    ght.setAngleStep(5)
    ght.setLevels(360)
    ght.setMinScale(0.5)
    ght.setMaxScale(1.5)
    ght.setScaleStep(0.05)
    ght.setAngleThresh(100)
    ght.setScaleThresh(100)
    ght.setPosThresh(100)
    ght.setAngleEpsilon(1)
    ght.setLevels(360)
    ght.setXi(10)

    # Load image
    print(path)

if __name__ == "__main__":
    main()