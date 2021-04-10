import time

import cv2
import torch

from bbox_visualizer import BBoxVisualizer
from deep_sort import DeepSort
from yolov5 import Detector
from threaded_video_capture import ThreadedVideoCapture
from tqdm import tqdm
from incremental_mean_tracker import IncrementalMeanTracker


def parse_detection(det):
    xywh = torch.empty(det.size(0), 4, dtype=det.dtype, device=det.device)
    xywh[:, :2] = (det[:, :2] + det[:, 2:4]) / 2
    xywh[:, 2:] = det[:, 2:4] - det[:, :2]
    confs = det[:, 4]
    return xywh, confs


def main():
    print('Connecting to camera')
    # cap = cv2.VideoCapture(0)
    cap = ThreadedVideoCapture('rtsp://admin:comvis@123@192.168.100.125:554/Streaming/Channels/101/')
    # cap = ThreadedVideoCapture('rtsp://admin:comvis@123@192.168.100.125/H264?ch=1&subtype=0')
    assert cap.isOpened(), 'Unable to connect to camera'
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = int(cap.get(cv2.CAP_PROP_FPS))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print('Loading models')
    detector = Detector('weights/yolov5s.pt', img_size=(640, 640),
                        conf_thresh=0.5, iou_thresh=0.5, agnostic_nms=False,
                        device=device)
    deepsort = DeepSort('weights/ckpt.t7',
                        max_dist=0.2, min_confidence=0.3,
                        nms_max_overlap=0.5, max_iou_distance=0.7,
                        max_age=100, lingering_age=5, n_init=5, nn_budget=100,
                        device=device)
    bboxes_visualizer = BBoxVisualizer()
    fps_estimator = IncrementalMeanTracker(max_count=cam_fps * 5)
    person_cls_id = detector.names.index('person')  # get id of 'person' class

    print(f'Starting capture, camera_fps={cam_fps}')
    # Start
    cap.start()
    win_name = 'MICA ReID Demo'
    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(win_name, width, height)
    frame_id = 0
    pbar = tqdm(desc=win_name)
    while True:
        start_it = time.time()
        ret, img = cap.read()
        if not ret:
            print('Unable to read camera')
            break
        detections = detector.detect([img])[0]

        num_people = 0
        if detections is not None:
            detections = detections[detections[:, -1].eq(person_cls_id)]  # filter person
            xywh, confs = parse_detection(detections)
            outputs = deepsort.update(xywh, confs, img)
            num_people = len(outputs)
            bboxes_visualizer.remove([t.track_id for t in deepsort.tracker.tracks
                                      if t.time_since_update > 3 or t.is_deleted()])
            bboxes_visualizer.update(outputs)
            # draw detections
            for pid in outputs[:, -1]:
                bboxes_visualizer.box(img, pid,
                                      label=f'Person {pid}',
                                      line_thickness=5,
                                      trail_trajectory=True,
                                      trail_bbox=False)
        # draw counting
        count_str = f'Number of people: {num_people}'
        img = bboxes_visualizer.text(img, count_str, (960, 25),
                                     fontScale=0.8, box_alpha=0.4,
                                     color=(255, 255, 255), box_color=(0, 0, 0))

        # show
        cv2.imshow(win_name, img)
        key = cv2.waitKey(1)
        elapsed_time = time.time() - start_it
        fps = fps_estimator.update(1 / elapsed_time)

        desc = f'[{frame_id:06d}] num_detections={num_people} fps={fps:.02f} elapsed_time={elapsed_time:.03f}'
        pbar.update()
        pbar.set_description(desc)
        # check key pressed
        if key == ord('q') or key == 27:  # q or esc to quit
            break
        elif key == ord('r'):  # r to reset tracking
            deepsort.reset()
            bboxes_visualizer.clear()
        elif key == 32:  # space to pause
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                break
        frame_id += 1
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
