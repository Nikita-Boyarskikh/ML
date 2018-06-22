import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_image', type=str, help='input image path', default='img.jpg')

    args = parser.parse_args()

    img = cv2.imread(args.in_image)
    img = cv2.medianBlur(img,5)
    gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gimg,cv2.HOUGH_GRADIENT,1,90,
                               param1=50,param2=20,minRadius=0,maxRadius=150)

    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv2.circle(img,(i[0],i[1]),i[2],(40,150,40),2)
        cv2.circle(img,(i[0],i[1]),2,(10,140,170),3)

    cv2.imshow('Circles',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
