import cv2
import argparse

def main():
    parser = argparse.ArgumentParser(description='Hough Transform')
    parser.add_argument('-i', '--image', type=str, required=True, help='Image path')
    args = parser.parse_args()

    img, detected_circles = get_circles(args.image)

    cv2.imwrite("circle_results.png", img)

def get_circles(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 100, 
                               param1=300, 
                               param2=100, 
                               minRadius=0, 
                               maxRadius=200
                               )
    circles = circles

    circle_imgs = []

    if circles is None:
        return img, circle_imgs

    for (x, y, r) in circles[0]:
        x = int(x)
        y = int(y)
        r = int(r * 1.1) + 10
        bounding_img = img[y-r:y+r, x-r:x+r]
        circle_imgs.append(bounding_img)
        # bounding_img = cv2.resize(bounding_img, dsize=(28, 28))
        # cv2.imwrite("sign_results.png", bounding_img)
        cv2.circle(img, (x, y), r, (0, 0, 255), 1)

    return img, circle_imgs

if __name__ == "__main__":
    main()