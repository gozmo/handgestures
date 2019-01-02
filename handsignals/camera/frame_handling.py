import cv2
import time

from handsignals.camera.record import start_capture
from handsignals.constants import Directories

class FrameHandling:
    def collect_data(self, frames = 100):
        captured_frames = start_capture(frames)
        self._save_frames_to_unlabel(captured_frames)

    def _save_frames_to_unlabel(self, captured_frames):
        print("saving frames: {}".format(captured_frames))
        for frame in captured_frames:
            self._save_frame(frame, Directories.UNLABEL)

    def _save_frame(self, frame, directory):
        name = time.time()
        filepath = "{}/{}.jpg".format(Directories.UNLABEL, name)
        print("writing file: {}".format(filepath))

        cv2.imwrite(filepath, frame)
