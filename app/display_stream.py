from pathlib import Path
import numpy as np
import cv2
import qimage2ndarray
from loguru import logger
from datetime import datetime

from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMainWindow

class Thread(QThread):
    change_pixmap = pyqtSignal(QImage)
    change_lamps = pyqtSignal(list)

    def __init__(self, parent: QMainWindow, wear_system, logger_filename: str = 'logging.conf'):
        """
        Args:
            parent: окно к которому относится поток.
            wear_system: детектор определения одежды.
            logger_filename: путь до файла логгирования.
        """
        super().__init__(parent)
        self.shoot_mode = parent.app_cfg['SHOOT_MODE']
        self.save_video_path = parent.app_cfg['SAVE_VIDEO_PATH']
        self.wear_system = wear_system
        if self.save_video_path != '':
            self.video_writer, self.video_writer_name = self.init_videowriter(parent.app_cfg['VIDEO_SAVE_FORMAT'],
                                                                              parent.app_cfg['VIDEO_FOURCC'],
                                                                              parent.app_cfg['VIDEO_NAME_PREFFIX'],
                                                                              parent.app_cfg['VIDEO_NAME_SUFFIX'])
        else:
            self.video_writer = None
        # self.devices_list = self.get_devices_list()

    def show_frame(self, frame: np.ndarray) -> None:
        """
        Отображает входной кадр.

        Args:
            frame: кадр для отображения.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = qimage2ndarray.array2qimage(frame)
        self.change_pixmap.emit(image)

    def init_videowriter(self, format_: str = 'mp4', fourcc: str = 'MP4V',
                         preffix_name: str = '', suffix_name: str = '') -> cv2.VideoWriter:
        """
        Инициализирует захватчик для видео.

        Args:
            format_: формат записи видео.
            fourcc: кодеки.
            preffix_name: префикс в начале названия записанного видео.
            suffix_name: суффикс в конце названия записанного видео (располгается по расширения).

        Returns:
            Захватчик для видео.
        """
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        input_fps = int(self.wear_system.loader.cap.get(cv2.CAP_PROP_FPS))
        width = int(
            self.wear_system.loader.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH))
        height = int(
            self.wear_system.loader.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT))

        output_name = preffix_name + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + \
            suffix_name + '.' + format_
        output_name = Path(self.save_video_path) / output_name
        video_writer = cv2.VideoWriter(str(output_name),
                                       fourcc, input_fps, (width, height))
        logger.info(f'Videowriter "{str(output_name)}" created')
        logger.debug(
            f'Videowriter params: fourcc={fourcc},fps={input_fps},width={width},height={height}')
        return video_writer, output_name

    def run(self, pipe: str = None, img_size: str = None) -> None:
        """
        Запускает обработку и отображение кадров.

        Args:
            pipe: видеопоток.
            img_size: размер изображения к которому привести.
        """
        if pipe is not None:
            self.wear_system.set_loader(pipe, img_size)
        if self.shoot_mode:
            for p, img, flip in self.wear_system.loader:
                self.show_frame(img)
                self.last_load = (img, flip)
                if self.video_writer is not None:
                    self.video_writer.write(img)
        else:
            for img, labels in self.wear_system.detect():
                self.show_frame(img)
                self.change_lamps.emit(labels)
                if self.video_writer is not None:
                    self.video_writer.write(img)

    @logger.catch()
    def stop(self):
        """
        Останавливает выполнение потока, освобождает видеопоток и останавливает захватчик видео.
        """
        logger.info(f'stop, source = {self.wear_system.loader.pipe}')
        self.wear_system.release()
        if self.video_writer is not None:
            self.video_writer.release()
