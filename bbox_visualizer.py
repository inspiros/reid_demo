import random

import cv2
import numpy as np
from matplotlib.cm import get_cmap


class BBoxVisualizer:
    def __init__(self, max_trail=10, num_colors=32, num_color_parts=8):
        self.bboxes = dict()
        self.max_trail = max_trail
        if self.max_trail < 0:
            self.max_trail = float('inf')

        self.num_colors = num_colors
        self.num_color_parts = num_color_parts
        self.num_color_per_part = self.num_colors // self.num_color_parts
        self.cmap = get_cmap('Spectral', num_colors)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_colors)]

    def update(self, track_outputs):
        for x1, y1, x2, y2, pid in track_outputs:
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            bbox = [x1, y1, x2, y2, xc, yc]
            if pid in self.bboxes:
                self.bboxes[pid].append(bbox)
                self.bboxes[pid] = self.bboxes[pid][-self.max_trail:]
            else:
                self.bboxes[pid] = [bbox]

    def remove(self, pids):
        for pid in pids:
            if pid in self.bboxes:
                del self.bboxes[pid]

    def clear(self):
        self.bboxes.clear()

    def __getitem__(self, item):
        return self.bboxes[item]

    def box(self, img, pid, label=None, label_color=(0, 0, 0),
            line_thickness=3, trail_trajectory=False, trail_bbox=False):
        # Plots one bounding box on image img
        color = np.asarray(
            self.cmap(pid // self.num_color_parts + (pid % self.num_color_parts) * self.num_color_per_part)[:3]
        ).__mul__(255).round().astype(np.int64).tolist()
        # self.colors[pid % len(self.colors)]
        bboxes = self.bboxes[pid]
        pc_ = None
        for bbox_id, bbox in enumerate(bboxes):
            p1, p2 = (round(bbox[0]), round(bbox[1])), (round(bbox[2]), round(bbox[3]))
            pc = (round(bbox[4]), round(bbox[5]))
            is_last_bbox = bbox_id == len(bboxes) - 1
            d_thickness = line_thickness if is_last_bbox else max(line_thickness - len(bboxes) + bbox_id, 1)
            if trail_trajectory:
                if bbox_id:
                    cv2.line(img, pc_, pc, color, thickness=d_thickness, lineType=cv2.LINE_AA)
                pc_ = pc
            if trail_bbox or is_last_bbox:
                cv2.rectangle(img, p1, p2, color, thickness=d_thickness, lineType=cv2.LINE_AA)
            if is_last_bbox and label is not None:
                font_thickness = max(line_thickness // 3, 1)
                font_scale = max(line_thickness / 10, 0.2)
                text_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=font_thickness)[0]
                p2 = p1[0] + text_size[0], p1[1] - text_size[1] - 3
                cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(img, label, (p1[0], p1[1] - 2), 0, font_scale, label_color,
                            thickness=font_thickness, lineType=cv2.LINE_AA)

    def text(self, img, text, xy, fontScale=1, thickness=1,
             color=(0, 0, 0), box_color=(255, 255, 255), box_alpha=0):
        text_size = cv2.getTextSize(text, 0, fontScale=fontScale, thickness=thickness)[0]
        if box_alpha == 1:
            cv2.rectangle(img, (xy[0] - 5, xy[1] - 5), (xy[0] + text_size[0] + 5, xy[1] + text_size[1] + 8),
                          box_color, -1)
        elif 0 < box_alpha < 1:
            overlay = img.copy()
            cv2.rectangle(overlay, (xy[0] - 5, xy[1] - 5), (xy[0] + text_size[0] + 5, xy[1] + text_size[1] + 8),
                          box_color, -1)
            img = cv2.addWeighted(overlay, box_alpha, img, 1 - box_alpha, 0)
        cv2.putText(img, text, (xy[0], xy[1] + text_size[1]), 0,
                    fontScale=fontScale, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return img
