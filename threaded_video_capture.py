import threading
from typing import Any

import cv2


class CountDownLatch:
    def __init__(self, count=1):
        self.count = count
        self._lock = threading.Condition()

    def count_down(self):
        self._lock.acquire()
        self.count -= 1
        if self.count == 0:
            self._lock.notifyAll()
        self._lock.release()

    def acquire(self):
        self._lock.acquire()
        while self.count > 0:
            self._lock.wait()
        self._lock.release()


class ThreadedVideoCapture:
    def __init__(self,
                 src: Any = 0):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.thread = threading.Thread(target=self.run)
        self._running = False
        self._lock = threading.Lock()
        self._started_lock = CountDownLatch()

    def start(self):
        self._running = True
        self.thread.start()

    def run(self):
        grabbed, self.frame = self.cap.read()
        if not grabbed:
            self._running = False
        self._started_lock.count_down()
        while self._running:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self._running = False
            self._lock.acquire()
            self.frame = frame
            self._lock.release()

    def stop(self):
        self._running = False
        if self.thread.is_alive():
            self.thread.join()

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def read(self):
        self._lock.acquire()
        self._started_lock.acquire()
        ret = self.frame
        self._lock.release()
        return ret is not None, ret

    def release(self):
        self._running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()
