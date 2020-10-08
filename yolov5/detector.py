from numpy import random

from .detection_helpers import *
from .models.experimental import attempt_load

__all__ = ['Detector']


class Detector:
    def __init__(self, weights, img_size=(640, 640), conf_thresh=0.4, iou_thresh=0.5, agnostic_nms=False, device='cpu'):
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = torch.device(device)
        self.agnostic_nms = agnostic_nms
        self.model = attempt_load(weights, map_location=self.device)
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        if self.device.type == 'cuda':  # warm-up
            self.model(torch.rand(1, 3, self.img_size[1], self.img_size[0], device=self.device))

    @torch.no_grad()
    def detect(self, im0s):
        imgs = []
        ratio_pads = []
        for im0 in im0s:
            # Convert
            img, gain, pad = letterbox(im0, new_shape=self.img_size)
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(np.expand_dims(img, 0))
            imgs.append(img)
            ratio_pads.append((gain, pad))
        imgs = torch.from_numpy(np.concatenate(imgs)).float().to(self.device).div(255)

        # Inference
        detections = self.model(imgs)[0].cpu()
        detections = non_max_suppression(detections, self.conf_thresh, self.iou_thresh, agnostic=self.agnostic_nms)
        # Process detections
        for i in range(len(detections)):  # detections per image
            im0 = im0s[i]
            det = detections[i]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(det[:, :4], imgs.shape[2:], im0.shape)
                detections[i] = det
        return detections


if __name__ == '__main__':
    import cv2
    detector = Detector(weights='yolov5s.pt', device='cuda:0')
    cap = cv2.VideoCapture(0)
    assert cap.isOpened()
    _, img = cap.read()
    imgs = [img]
    cap.release()

    outs = detector.detect(imgs)
    for img, out in zip(imgs, outs):
        if out is not None:
            for x1, y1, x2, y2, conf, cls in out:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # cv2.putText(img, ''.join(out[:, -1].int().numpy().astype(str).tolist()), (5, 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('test', cv2.resize(img, (1020, 840)))
        cv2.waitKey(0)
    cv2.destroyAllWindows()
