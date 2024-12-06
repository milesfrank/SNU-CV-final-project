import cv2

def main() -> None:
    img_path = "scene.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20, 
                               param1=300, 
                               param2=200, 
                               minRadius=0, 
                               maxRadius=0
                               )
    circles = circles

    print(circles[0])

    for (x, y, r) in circles[0]:
        x = int(x)
        y = int(y)
        r = int(r)
        bounding_img = img[y-r:y+r, x-r:x+r]
        bounding_img = cv2.resize(bounding_img, dsize=(28, 28))
        cv2.imwrite("sign_results.png", bounding_img)
        cv2.circle(img, (x, y), r, (0, 0, 255), 1)

    cv2.imwrite("circle_results.png", img)

if __name__ == "__main__":
    main()