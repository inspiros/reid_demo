import cv2
from threaded_video_capture import ThreadedVideoCapture


def main():
    # cap = cv2.VideoCapture('rtsp://admin:comvis@123@172.16.90.125/H264?ch=1&subtype=0')
    cap = ThreadedVideoCapture('rtsp://admin:comvis@123@172.16.90.125/H264?ch=1&subtype=0')
    cap.start()
    assert cap.isOpened()

    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('rstp', img)
        else:
            print('noret')
        key = cv2.waitKey(1)
        if key == ord('q'):
            cap.release()
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
