import cv2
import numpy as np

def preprocess_image(img, save=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, dsize=(28, 28))
    if save:
        cv2.imwrite("resized.png", img)
    img = img[8:-8, 8:-8]
    if save:
        cv2.imwrite("cropped.png", img)
    img = np.where(img < 100, 0, 255)
    if save:
        cv2.imwrite("preprocessed.png", img)
    img = img.flatten()
    return img