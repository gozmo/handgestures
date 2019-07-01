import cv2
import time

from handsignals.camera.record import start_capture
from handsignals.constants import Directories


class FrameHandling:
    def collect_data(self, frames=100):
        captured_frames = start_capture(frames)
        files = self._save_frames_to_unlabel(captured_frames)
        return files

    def _save_frames_to_unlabel(self, captured_frames):
        files = []
        for frame in captured_frames:
            filename = self._save_frame(frame, Directories.UNLABEL)
            files.append(filename)
        return files

    def _save_frame(self, frame, directory):
        name = time.time()
        filepath = "{}/{}.jpg".format(Directories.UNLABEL, name)
        print("writing file: {}".format(filepath))

        cv2.imwrite(filepath, frame)
        return filepath
