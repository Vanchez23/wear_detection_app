import sys
sys.path.append('app')
import os
import copy
from pathlib import Path
import yaml
import numpy as np
import cv2

from PyQt5.QtWidgets import QApplication

from display_stream import Thread
from wearing_detector import WearingDetector
from app import MainWindow

class TestDisplayStream:

    def setup(self,):
        app_cfg_path = Path('app/app_config.yaml')
        assert app_cfg_path.exists(), f'config file {app_cfg_path} doesn\'t exists'

        cfg_path = Path('wearing_detector/configs/wearing_detector_config.yaml')
        assert cfg_path.exists(), f'config file {cfg_path} doesn\'t exists'

        with open(app_cfg_path) as f:
            app_cfg = yaml.safe_load(f)

        old_app_cfg = copy.deepcopy(app_cfg)
        app_cfg['VIDEO_NAME_PREFFIX'] = 'test_'
        app_cfg['VIDEO_NAME_SUFFIX'] = '_test'

        with open(app_cfg_path, 'w') as f:
            yaml.dump(app_cfg, f)

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        try:
            self.app = QApplication(sys.argv)
            self.window = MainWindow(Path('app/ui/main.ui'),
                                     Path('app/ui/select_stream.ui'),
                                     Path('app/ui/image_window.ui'),
                                     app_cfg_name=app_cfg_path,
                                     wearing_detector_cfg_name=Path('wearing_detector/configs/wearing_detector_config.yaml'))

            self.detect_system = WearingDetector(cfg)
            self.stream = Thread(self.window, self.detect_system)
        except Exception as ex:
            with open(app_cfg_path, 'w') as f:
                yaml.dump(old_app_cfg)
            raise ex

    def test_videowriter(self, seconds=60):

        video_writer = self.stream.video_writer
        path = str(self.stream.video_writer_name)
        input_fps = int(self.detect_system.loader.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.detect_system.loader.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.detect_system.loader.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT))
        assert video_writer.isOpened()

        frames = np.random.randint(0, 256,
                                  size=[input_fps*seconds, height,width,3], dtype=np.uint8)
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

        cap = cv2.VideoCapture(path)
        assert cap.isOpened()

        ret_val, frame = cap.read()
        assert ret_val

        cap.release()
        os.remove(path)

    def __del__(self):
        self.stream.stop()
        self.stream.quit()
        self.window.close()
        self.app.quit()
