from pathlib import Path
import yaml
from loguru import logger

import numpy as np
from typing import Dict, List

from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSlot, Qt

from select_stream import SelectStreamWindow
from show_image import ImageWindow
from display_stream import Thread
from wearing_detector import WearingDetector

class MainWindow(QMainWindow):
    def __init__(self, main_ui_name: (str, Path), sel_camera_ui_name: (str, Path),
                 img_window_ui_name: (str, Path), app_cfg_name: (str, Path),
                 wearing_detector_cfg_name: (str, Path)):
        """
        
        Args:
            main_ui_name: xml разметка интерфейса главного окна приложения.
            sel_camera_ui_name: xml разметка интерфейса окна выбора видеопотока.
            img_window_ui_name: xml разметка интерфейса окна отображения распознавания (в shoot_mode режиме).
            app_cfg_name: путь до конфигурации приложения.
            wearing_detector_cfg_name: путь до конфигурации детектора комплектности одежды.
        """

        super(MainWindow, self).__init__()
        self.main_ui_name = main_ui_name
        self.sel_camera_ui_name = sel_camera_ui_name
        self.img_window_ui_name = img_window_ui_name

        # Initialize
        with open(app_cfg_name) as f:
            self.app_cfg = yaml.load(f, yaml.FullLoader)
        with open(wearing_detector_cfg_name) as f:
            self.wearing_detector_cfg = yaml.load(f, yaml.FullLoader)
        if 'LOGGER_FILENAME' in self.app_cfg:
            self.set_logger(self.app_cfg['LOGGER_FILENAME'], self.app_cfg.get('LOGGER_LEVEL','INFO'))
        self.init_ui(self.main_ui_name)
        self.stream_window = SelectStreamWindow(self.sel_camera_ui_name)

    @pyqtSlot(QImage)
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

    @pyqtSlot(list)
    def set_lamps_colors(self, detected_clothes: List[str]) -> None:
        """
        Задает цвета индикаторов наличия одежды.

        Args:
            detected_clothes: распознанная одежда.
        """
        for clothes, lamp in self.lamps.items():
            
            # if clothes == self.equipped_name and all([bool(v) for k, v in self.wait_colors.items() if k != self.equipped_name]):
            #     self.wait_colors[clothes] = self.wait_const
            #     lamp.setStyleSheet(self.GREEN_LIGHT_STYLE)
            # else:
            #     lamp.setStyleSheet(self.RED_LIGHT_STYLE)
            #
            # if clothes in detected_clothes:
            #     self.wait_colors[clothes] = self.wait_const
            #     lamp.setStyleSheet(self.GREEN_LIGHT_STYLE)
            # elif self.wait_colors[clothes] > 0:
            #     self.wait_colors[clothes] -= 1
            #     lamp.setStyleSheet(self.GREEN_LIGHT_STYLE)
            # else:
            #     lamp.setStyleSheet(self.RED_LIGHT_STYLE)
            if clothes == self.equipped_name:
                if len(detected_clothes) == len(self.lamps) - 1:
                    lamp.setStyleSheet(self.GREEN_LIGHT_STYLE)
                else:
                    lamp.setStyleSheet(self.RED_LIGHT_STYLE)
            else:
                if clothes in detected_clothes:
                    lamp.setStyleSheet(self.GREEN_LIGHT_STYLE)
                else:
                    lamp.setStyleSheet(self.RED_LIGHT_STYLE)



    def set_logger(self, logger_filename: str, logger_level='INFO') -> None:
        """
        Задает логгирование приложения.

        Args:
            logger_filename: путь до файла логгирования.
            logger_level: уровень логирования.
        """
        if logger_filename.endswith('.log'):
            format_ = "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} - <level>{message}</level>',"
            logger.add(logger_filename,
                       format=format_,
                       level=logger_level)

    @logger.catch()
    def run_thread(self, pipe: (str, int) = None) -> None:
        """
        Запускает видеопоток.

        Args:
            pipe: видеопоток.
        """
        if hasattr(self, 'stream'):
            self.closeEvent(None)
        if hasattr(self, 'detect_system'):
            self.detect_system.release()
        self.detect_system = WearingDetector(self.wearing_detector_cfg, pipe)
        self.stream = Thread(self, self.detect_system)
        self.stream.change_pixmap.connect(self.set_image)
        self.stream.change_lamps.connect(self.set_lamps_colors)
        self.stream.start()
        logger.debug('run thread')

    def select_camera(self) -> None:
        """
        Реализует выбор видеопотока и его запуск.
        """
        apply = self.stream_window.exec_()
        if not apply:
            return

        if self.stream_window.useIPCamera.isChecked():
            # 'rtsp://username:password@192.168.1.64/1'
            # 'rtsp://admin:camera12345@172.22.101.169/1'
            source = self.stream_window.connectString.text()
        elif self.stream_window.useWebCamera.isChecked():
            source = 0
        elif self.stream_window.useVideo.isChecked():
            if self.stream_window.video_path == '':
                return
            source = self.stream_window.video_path
        else:
            source = 0
            logger.error('None of the options are checked')

        self.run_thread(source)

    @logger.catch()
    def recognize_image(self, *args) -> None:
        """
        Распознает последний кадр из видеопотока.
        """
        # image,labels = self.detect_system.detect_once()
        return 
        image, labels = self.detect_system.detect_once(*self.stream.last_load)
        
        self.image_window = ImageWindow(
            self.img_window_ui_name,
            self.app_cfg['LAMP_NAMES'])
        self.image_window.show_frame(image)
        self.image_window.set_lamps_colors(labels)
        self.image_window.show()
        logger.info(f'detected labels = {labels}')
        logger.info('show image window')

    def init_ui(self, ui_name: str ) -> None:
        """
        Инициализаци элементов интерфейса приложения.

        Args:
            ui_name: xml разметка интерфейса главного окна приложения.
        """
        uic.loadUi(ui_name, self)
        self.choose_stream_button.clicked.connect(self.select_camera)
        self.GREEN_LIGHT_STYLE = 'background-color: green;border: 1px solid black;'
        self.RED_LIGHT_STYLE = 'background-color: red;border: 1px solid black;'
        self.equipped_name = self.app_cfg['LAMP_NAMES'][-1]
        self.lamps = {
            lamp_name: getattr(
                self,
                'lamp_' +
                lamp_name) for lamp_name in self.app_cfg['LAMP_NAMES']}
        self.wait_const = 25
        self.wait_colors = {lamp_name: self.wait_const  for lamp_name in self.app_cfg['LAMP_NAMES']}
        if self.app_cfg['SHOOT_MODE']:
            for lamp_name, lamp in self.lamps.items():
                if hasattr(self, 'label_' + lamp_name):
                    getattr(self, 'label_' + lamp_name).hide()
                elif hasattr(self, 'label_' + lamp_name[:-1] + 's'):
                    getattr(self, 'label_' + lamp_name[:-1] + 's').hide()
                lamp.hide()
            self.shoot_button.clicked.connect(self.recognize_image)
        else:
            self.shoot_button.hide()
        if self.app_cfg['SAVE_VIDEO_PATH'] is not None and self.app_cfg['SAVE_VIDEO_PATH'] != '':
            Path(
                self.app_cfg['SAVE_VIDEO_PATH']).mkdir(
                parents=True,
                exist_ok=True)
            self.lamp_record.show()
            self.label_record.show()
        else:
            self.lamp_record.hide()
            self.label_record.hide()
        self.set_lamps_colors([])

    def closeEvent(self, event) -> None:
        """
        Останавливает видеопоток перед закрытием приложения.
        """
        try:
            self.stream.stop()
            self.stream.quit()
        except Exception as ex:
            logger.error(ex)


if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path
    import torch
    logger.info(f'cuda is available = {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn')
    logger.info(f'workdir = {os.getcwd()}')
    # because yolo model need that file structure
    sys.path.append('wearing_detector/yolov5')
    app = QApplication(sys.argv)
    window = MainWindow(Path('app/ui/main.ui'),
                        Path('app/ui/select_stream.ui'),
                        Path('app/ui/image_window.ui'),
                        app_cfg_name=Path('app/app_config.yaml'),
                        wearing_detector_cfg_name=Path('wearing_detector/configs/wearing_detector_config.yaml'))
    window.run_thread()
    window.show()
    sys.exit(app.exec_())
