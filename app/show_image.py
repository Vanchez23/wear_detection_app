from typing import List
import cv2
import qimage2ndarray
import numpy as np

from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt


class ImageWindow(QWidget):
    def __init__(self, ui_name: str, lamp_names: str):
        """
        Args:
            ui_name: xml разметка интерфейса окна отображения распознавания (в shoot_mode режиме).
            lamp_names: имена индикаторов наличия одежды.
        """
        super(ImageWindow, self).__init__()
        self.ui_name = ui_name
        self.lamp_names = lamp_names
        self.init_ui(self.ui_name)

    def show_frame(self, frame: np.ndarray) -> None:
        """
        Отображает входной кадр.

        Args:
            frame: кадр для отображения.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = qimage2ndarray.array2qimage(frame)
        self.set_image(image)

    def set_image(self, image: np.ndarray) -> None:
        """
        Задает отображаемый кадр.

        Args:
            image: кадр из видеопотока.
        """
        pixmap = QPixmap.fromImage(image)
        pixmap = pixmap.scaled(
            self.screen.width(),
            self.screen.height(),
            Qt.KeepAspectRatio)
        self.screen.setPixmap(pixmap)
        self.screen.setAlignment(Qt.AlignCenter)

    def set_lamps_colors(self, detected_clothes: List[str]) -> None:
        """
        Задает цвета индикаторов наличия одежды.

        Args:
            detected_clothes: распознанная одежда.
        """
        for clothes, lamp in self.lamps.items():
            if clothes == self.equipped_name:
                if detected_clothes == len(self.lamps) - 1:
                    lamp.setStyleSheet(self.GREEN_LIGHT_STYLE)
                else:
                    lamp.setStyleSheet(self.RED_LIGHT_STYLE)
            else:
                if clothes in detected_clothes:
                    lamp.setStyleSheet(self.GREEN_LIGHT_STYLE)
                else:
                    lamp.setStyleSheet(self.RED_LIGHT_STYLE)

    def init_ui(self, ui_name: str) -> None:
        """
        Инициализаци элементов интерфейса приложения.

        Args:
            ui_name: xml разметка интерфейса главного окна приложения.
        """
        uic.loadUi(ui_name, self)
        self.GREEN_LIGHT_STYLE = 'background-color: green;border: 1px solid black;'
        self.RED_LIGHT_STYLE = 'background-color: red;border: 1px solid black;'
        self.equipped_name = self.lamp_names[-1]
        self.lamps = {
            lamp_name: getattr(
                self,
                'lamp_' +
                lamp_name) for lamp_name in self.lamp_names}
        self.set_lamps_colors([])
