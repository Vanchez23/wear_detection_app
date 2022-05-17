import sys

from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.uic import loadUi

# if sys.platform == 'win32':
# import win32com.client
# elif sys.platform == 'linux':
# import re
# import subprocess
# elif sys.platform == 'darwin':
# pass


class SelectStreamWindow(QDialog):
    def __init__(self, ui_name: str) -> None:
        """
        Args:
            ui_name: xml разметка интерфейса окна выбора видеопотока.
        """
        super(SelectStreamWindow, self).__init__()
        loadUi(ui_name, self)
        self.useIPCamera.clicked.connect(self.use_ip_camera)
        self.useWebCamera.clicked.connect(self.use_webcamera)
        self.useVideo.clicked.connect(self.use_video)
        self.selectVideo.clicked.connect(self.choose_video)
        self.connectString.setEnabled(True)
        self.cameraIndexes.setEnabled(False)
        self.selectVideo.setEnabled(False)

    def use_ip_camera(self) -> None:
        """
        Изменяет радиокнопку при выборе ip камеры.
        """
        self.connectString.setEnabled(True)
        self.cameraIndexes.setEnabled(False)
        self.selectVideo.setEnabled(False)

    def use_webcamera(self) -> None:
        """
        Изменяет радиокнопку при выборе веб-камеры.
        """
        self.connectString.setEnabled(False)
        self.cameraIndexes.setEnabled(True)
        self.selectVideo.setEnabled(False)

    def use_video(self) -> None:
        """
        Изменяет радиокнопку при выборе видео.
        """
        self.connectString.setEnabled(False)
        self.cameraIndexes.setEnabled(False)
        self.selectVideo.setEnabled(True)

    def choose_video(self) -> str:
        """
        Открыть диалоговое окно при выборе видеопотока.
        """
        self.video_path = QFileDialog.getOpenFileName()[0]
        return self.video_path

    # def get_devices_list(self):
    #     """
    #     Получить список доступных веб-камер.
    #     """
    #     if sys.platform == 'win32':
    #         wmi = win32com.client.GetObject('winmgmts:')
    #         for usb in wmi.InstancesOf('Win32_USBHub'):
    #             print(usb.DeviceID)
    #     elif sys.platform == 'linux':
    #         pass
    #     elif sys.platform == 'darwin':
    #         pass
