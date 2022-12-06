import cv2
import time

def main():
    cap = cv2.VideoCapture('../video/BanTrac.mp4')
    # Resolution 640*480
    time.sleep(1)
    if cap is None or not cap.isOpened():
        print('Khong the mo file video')
        return
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE);
    n = 1
    count = 200
    while True:
        [success, img] = cap.read()
        ch = cv2.waitKey(30)
        if success:
            img = cv2.flip(img, 0)
            img = cv2.rotate(img, cv2.ROTATE_180)
            imgROI = img[40:(40+480),:] # Tao ra anh 480x480

            imgROI = cv2.resize(imgROI,(320,320))
            cv2.imshow('Image', imgROI)
        else:
            break
        if n%4 == 0:
            filename = '../image/BanTrac/BanTrac%04d.bmp'%(count)
            cv2.imwrite(filename,imgROI)
            count = count + 1
        n = n + 1
    return

if __name__ == "__main__":
    main()
