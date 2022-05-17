import sys
sys.path.append('app')
from pathlib import Path

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication

from app import MainWindow

class TestApp:
    def setup(self):
        self.app = QApplication(sys.argv)
        self.window = MainWindow(Path('app/ui/main.ui'),
                            Path('app/ui/select_stream.ui'),
                            Path('app/ui/image_window.ui'),
                            app_cfg_name=Path('app/app_config.yaml'),
                            wearing_detector_cfg_name=Path('wearing_detector/configs/wearing_detector_config.yaml'))

        self.classes = self.window.app_cfg['LAMP_NAMES']
        assert self.classes

    def get_lamps_colors(self):

        styles = dict()
        for lamp_name, lamp in self.window.lamps.items():
            if lamp_name != self.window.equipped_name:
                styles[lamp_name] = lamp.styleSheet() == self.window.GREEN_LIGHT_STYLE
        return styles

    def check_lamps(self, classes=None):
        if classes is not None:
            self.window.set_lamps_colors(classes)
        else:
            classes = []
        styles = self.get_lamps_colors()
        for lamp_name, is_green in styles.items():
            if lamp_name in classes:
                assert is_green, f'Lamp {lamp_name} must be green'
            else:
                assert not is_green, f'Lamp {lamp_name} must be red'

    def test_lamps_colors(self):
        self.check_lamps()
        self.check_lamps(self.classes)
        self.check_lamps(self.classes[:3])
        self.check_lamps(self.classes[:-1])

    def test_select_camera(self, qtbot, monkeypatch):

        qtbot.addWidget(self.window.stream_window)

        qtbot.mouseClick(self.window.stream_window.useVideo, QtCore.Qt.LeftButton)
        assert not self.window.stream_window.useIPCamera.isChecked()
        assert not self.window.stream_window.useWebCamera.isChecked()
        assert self.window.stream_window.useVideo.isChecked()

        qtbot.mouseClick(self.window.stream_window.useIPCamera, QtCore.Qt.LeftButton)
        assert self.window.stream_window.useIPCamera.isChecked()
        assert not self.window.stream_window.useWebCamera.isChecked()
        assert not self.window.stream_window.useVideo.isChecked()

        qtbot.mouseClick(self.window.stream_window.useWebCamera, QtCore.Qt.LeftButton)
        assert not self.window.stream_window.useIPCamera.isChecked()
        assert self.window.stream_window.useWebCamera.isChecked()
        assert not self.window.stream_window.useVideo.isChecked()

        accept_calls = []
        monkeypatch.setattr(self.window.stream_window,
                            "accept", lambda: accept_calls.append(1))
        self.window.stream_window.accept()
        assert accept_calls == [1]
        assert not self.window.stream_window.useIPCamera.isChecked()
        assert self.window.stream_window.useWebCamera.isChecked()
        assert not self.window.stream_window.useVideo.isChecked()
