import os
import time
import numpy as np
import copy
import cv2
import pandas as pd
from loguru import logger
import multiprocessing as mp
from typing import List, Dict, Tuple

from .yolo_model import YoloModel
from .hrnet_model import HRNetModel
from .wear_control import Skeleton
from .constants import person_detection_classes, wear_detection_classes
from wearing_detector.utils.datasets import LoadWebcam
from wearing_detector.utils.general import plot_one_box
from wearing_detector.utils.utils_hrnet import draw_joints, draw_points, COLORS, LINK_PAIRS, POINT_COLOR


class Detector:
    def __init__(self, cfg: Dict):
        """
        Класс с основными моделями (Person Detection, Clothes Detection, Keypoint Detection).
        Содержит методы загрузки и инициализации моделей, методы для предикта и 
        метод наложения ключевых точек на одежду.
        Args:
            cfg: общий конфиг.
        """
        self._config = cfg
        self.common_cfg = self._config['VIDEOSTREAM']
        self.person_model_cfg = self._config['PERSON_MODEL']
        self.clothes_model_cfg = self._config['CLOTHES_MODEL']
        self.keypoints_model_cfg = self._config['KEYPOINTS_MODEL']
        self.wear_control_cfg = self._config['WEAR_CONTROL']

        self.set_person_model(self.person_model_cfg)
        self.set_clothes_model(self.clothes_model_cfg)
        try:
            self.keypoints_model = HRNetModel(self.keypoints_model_cfg)
            logger.info(
                f'keypoints model loaded from {self.keypoints_model_cfg["WEIGHTS"]}')
        except Exception as ex:
            logger.error(ex)

    @logger.catch()
    def set_clothes_model(self, clothes_config: Dict):
        """
        Инициализирует модель по определению одежды.

        Args:
            clothes_config: конфигурации для модели по определению одежды.
        """
        self.clothes_model = YoloModel(clothes_config['WEIGHTS'],
                                       clothes_config['DEVICE'],
                                       clothes_config['IMG_SIZE'],
                                       clothes_config['CONF_THRES'],
                                       clothes_config['IOU_THRES'],
                                       clothes_config['AGNOSTIC_NMS'],
                                       clothes_config['PADDING'],
                                       make_preprocess=True)
        logger.info(f'clothes model loaded from {clothes_config["WEIGHTS"]}')
        self._clothes_labels = self.clothes_model.classes
        self._clothes_labels.append('gloves')
        
        self._clothes_colors = {
            name: [
                np.random.randint(
                    0,
                    255) for _ in range(3)] for name in self._clothes_labels}

    @logger.catch()
    def set_person_model(self, person_config: Dict):
        """
        Инициализирует модель по определению человека.

        Args:
            person_config: конфигурации для модели по определению человека.
        """
        self.person_model = YoloModel(person_config['WEIGHTS'],
                                      person_config['DEVICE'],
                                      person_config['IMG_SIZE'],
                                      person_config['CONF_THRES'],
                                      person_config['IOU_THRES'],
                                      person_config['AGNOSTIC_NMS'],
                                      person_config['PADDING'],
                                      make_preprocess=True)
        logger.info(f'person model loaded from {person_config["WEIGHTS"]}')
        
        self._person_colors = {
            name: [
                np.random.randint(
                    0, 255) for _ in range(3)] for name in self.person_model.classes}

    def person_detect(self, img: np.ndarray, img0_shape: List,
                      img_name: str = '') -> List[Dict]:
        """
        Детектирует людей на изображении.

        Args:
            img: изображение для детекции.
            img0_shape: размер оригинального изображения.
            img_name: название изображения.

        Returns:
            Список с детекциями по каждому найденному человеку.
        """
        yolo_preds = self.person_model(img, img0_shape)
        person_preds = []
        for pred in yolo_preds:
            if pred['label'] == 'person':
                pred['image_name'] = img_name
                pred['coords'] = [int(pred['x1']), int(
                    pred['y1']), int(pred['x2']), int(pred['y2'])]
                person_preds.append(pred)
        return person_preds

    def clothes_detect(self, img: np.ndarray, img0_shape: List,
                       img_name: str, bbox: List) -> Tuple[List, List]:
        """
        Детектирует одежду на изображении.

        Args:
            img: изображение для детекции.
            img0_shape: размер оригинального изображения.
            img_name: название изображения.
            bbox: прямоугольник в котором детектируется одежда.

        Returns:
            Список с детекциями по каждой одежде.
        """
        labels = []
        count_gloves = 0

        clothes_preds = self.clothes_model(img, img0_shape, bbox)
        for pred in clothes_preds:
            pred['image_name'] = img_name
            if pred['label'] == 'gloves':
                if count_gloves == 0:
                    labels.append('glove1')
                elif count_gloves == 1:
                    labels.append('glove2')
                count_gloves += 1
            else:
                labels.append(pred['label'])
        return clothes_preds, labels

    def keypoints_detect(self, img: np.ndarray, bbox: List,
                         flip: bool = False) -> Dict[str, Dict]:
        """
        Детектирует ключевые точки на изображении.

        Args:
            img: изображение для детекции.
            bbox: прямоугольник в котором детектируется одежда.
            flip: флаг для отражения названий левых и правых ключевых точек.

        Returns:
            Словарь с координатами по каждой ключевой точке.
        """
        coords, confs = self.keypoints_model(img, bbox)
        keypoints = dict()
        for pred, conf, idx in zip(coords, confs, range(
                len(self.keypoints_model_cfg['KEYPOINTS_NAMES']))):
            pred = list(pred)
            label = self.keypoints_model_cfg['KEYPOINTS_NAMES'][idx]
            if flip:
                if label.startswith('l_'):
                    label = label.replace('l_', 'r_')
                elif label.startswith('r_'):
                    label = label.replace('r_', 'l_')
            keypoints[idx] = {'point': (int(pred[0]), int(pred[1])),
                              'conf': conf,
                              'label': label}
        return keypoints

    def wear_control(self, clothes_preds: List,
                     keypoints: Dict) -> pd.DataFrame:
        """
        Определяет находится ли одежда в районе соответсвующих ключевых точек человека.

        Args:
            clothes_preds: предсказания одежды.
            keypoints: предсказанные ключевые точки.

        Returns:
            Датафрейм с определением по каждой одежде находится ли она на человеке.
        """
        skelet = Skeleton(keypoints)
        preds_df = pd.DataFrame.from_dict(clothes_preds)
        for name in ('result', 'left_result', 'right_result'):
            preds_df[name] = None

        for idx, pred in preds_df.iterrows():
            result = None
            if pred['label'] == 'jacket':
                score, _ = skelet.torso_inside(
                    pred[['x1', 'y1', 'x2', 'y2']], self.wear_control_cfg['MARGINS']['jacket'])
                result = score >= self.wear_control_cfg['THRESHOLDS']['jacket']
            elif pred['label'] == 'pants':
                score, _ = skelet.legs_inside(
                    pred[['x1', 'y1', 'x2', 'y2']], self.wear_control_cfg['MARGINS']['pants'])
                result = score >= self.wear_control_cfg['THRESHOLDS']['pants']
            elif pred['label'] == 'gloves':
                gloves_result = 0
                for side in ('left', 'right'):
                    score, point_inside = skelet.hand_inside(pred[['x1', 'y1', 'x2', 'y2']], side=side,
                                                             margin=self.wear_control_cfg['MARGINS']['gloves'])
                    result = score >= self.wear_control_cfg['THRESHOLDS']['gloves']
                    if not result:
                        score, dist = skelet.wrist_near(
                            pred[['x1', 'y1', 'x2', 'y2']], side=side)
                        result = score >= self.wear_control_cfg['THRESHOLDS']['gloves']

                    pred[side + '_result'] = result
                    gloves_result += result

                result = gloves_result > 0
            elif pred['label'] == 'shield':
                score, other = skelet.head_inside(pred[['x1', 'y1', 'x2', 'y2']],
                                                  margin=self.wear_control_cfg['MARGINS']['shield'])
                result = score >= self.wear_control_cfg['THRESHOLDS']['shield']

                if not result:
                    score, other = skelet.head_near(pred[['x1', 'y1', 'x2', 'y2']],
                                                    eye_nose_dist_coef=2., repeat_eye_nose_dist_coef=5,
                                                    min_eye_nose_dist=0.,
                                                    ears_dist_coef=5., repeat_ears_dist_coef=5, min_ears_dist=0.)
                    result = score >= self.wear_control_cfg['THRESHOLDS']['shield']
            elif pred['label'] == 'helmet':
                score, other = skelet.head_inside(pred[['x1', 'y1', 'x2', 'y2']],
                                                  margin=self.wear_control_cfg['MARGINS']['helmet'])
                result = score >= self.wear_control_cfg['THRESHOLDS']['helmet']
                if not result:
                    score, other = skelet.head_near(pred[['x1', 'y1', 'x2', 'y2']],
                                                    eye_nose_dist_coef=2., repeat_eye_nose_dist_coef=2,
                                                    min_eye_nose_dist=0.,
                                                    ears_dist_coef=3., repeat_ears_dist_coef=3, min_ears_dist=0.)
                    result = score >= self.wear_control_cfg['THRESHOLDS']['helmet']

            pred['result'] = result
        return preds_df

    def filtering(self, preds_df: pd.DataFrame) -> List[str]:
        """
        Фильтрует результаты определения одежды на ключевых точках.

        Args:
            preds_df: датафрейм с определением по каждой одежде находится ли она на человеке.

        Returns:
            Список одежды, которая находится на человеке.
        """
        labels = []
        if 'label' in preds_df.columns:
            preds_df_g = preds_df.groupby('label').sum()
            for label, row in preds_df_g.iterrows():
                if row['result']:
                    if label != 'gloves':
                        labels.append(label)
                    else:
                        labels.append('glove1')
                        if row['result'] > 1:
                            labels.append('glove2')

        return labels

    def draw(self, img, person_preds: List = [], clothes_preds: List = [],
             keypoints_preds: Dict = {}) -> np.ndarray:
        """
        Рисует bounding box человека, одежды и ключевые точки.

        Args:
            img: изображение для отрисовки.
            person_preds: список с детекциями по каждому найденному человеку.
            clothes_preds: список с детекциями по каждой одежде.
            keypoints_preds: cловарь координат по каждой ключевой точке.

        Returns:
            Изображение с отрисованными детекциями.
        """
        if person_preds:
            for pred in person_preds:
                plot_one_box([pred['x1'], pred['y1'], pred['x2'], pred['y2']],
                             img, self._person_colors[pred['label']], pred['label'], 3)
        if clothes_preds:
            for pred in clothes_preds:
                plot_one_box([pred['x1'], pred['y1'], pred['x2'], pred['y2']],
                             img, self._clothes_colors[pred['label']], pred['label'], 3)
        if keypoints_preds:
            draw_joints(img, keypoints_preds, LINK_PAIRS, COLORS)
            draw_points(img, keypoints_preds, POINT_COLOR, False, 5)
        return img

    def stages_flow(self, img0: np.ndarray, flip: bool,
                    path: str = '') -> Tuple[List, List, Dict, List, Dict]:
        """
        Объединяет все стадии детекции.

        Args:
            img0: оригинал кадра.
            flip: флаг для отражения.
            path: путь до изображения.

        Returns:
            Предсказания по стадиям детекции человека, одежды и ключевых точек, а также список одежды на человеке.
        """
        clothes_preds, keypoints, labels = [], [], []
        start_person_detect = time.time()
        person_preds = self.person_detect(
            img0, img0.shape, os.path.basename(path))
        person_detect_time = time.time() - start_person_detect

        clothes_detect_time = 0
        keyp_detect_time = 0
        wear_control_time = 0
        for person in person_preds:
            start_clothes_detect = time.time()
            clothes_preds, labels = self.clothes_detect(
                img0, img0.shape, os.path.basename(path), person['coords'])
            clothes_detect_time += time.time() - start_clothes_detect

            # if len(self._clothes_labels) == len(labels):
            # Специально чтобы модель отрабатывала постоянно
            if True:
                start_keyp_detect = time.time()
                keypoints = self.keypoints_detect(img0, person['coords'], flip)
                keyp_detect_time = time.time() - start_keyp_detect

                start_wear_control = time.time()
                preds_df = self.wear_control(clothes_preds, keypoints)
                labels = self.filtering(preds_df)
                wear_control_time = time.time() - start_wear_control
            else:
                labels = []

        clothes_detect_time /= max(len(person_preds), 1)
        meta = {'person_detect_time': person_detect_time,
                'clothes_detect_time': clothes_detect_time,
                'keyp_detect_time': keyp_detect_time,
                'wear_control_time': wear_control_time}
        return person_preds, clothes_preds, keypoints, labels, meta

    def detect_once(self, img0: np.ndarray = None,
                    flip: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Детектировать один кадр

        Args:
            img0: оригинальное изображение.
            flip: отразить названия сторон найденных ключевых точек.

        Returns:
            Входное изображение с нарисованными детекциями(возможно и без детекций, зависит от конфига)
            и список одежды на человеке.
        """
        logger.info('detect shooted image')
        t0 = time.time()

        if img0 is None:
            _, img0, flip = next(self.loader)

        (last_person_preds,
         last_clothes_preds,
         last_keypoints_preds,
         labels,
         meta) = self.stages_flow(img0, flip)

        if self.common_cfg['DRAW_LABELS']:
            self.draw(
                img0,
                last_person_preds,
                last_clothes_preds,
                last_keypoints_preds)

        logger.info(
            f'image recognized, detection time = {(time.time() - t0):.3} sec')
        return img0, labels


class Artist:
    """
    Класс для отрисовки боксов. Запоминает предыдущие боксы и отрисовывает их,
    елси приходит пустой список.
    Args:
        person_classes: List of str, список классов для персон детекшн модели
        clothes_classes: List of str, список классов для модели одежды
    """
    def __init__(self, person_classes, clothes_classes):
        self._person_colors = {
            name: [np.random.randint(0, 255) for _ in range(3)] 
            for name in person_classes
        }
        
        self._clothes_labels = clothes_classes
        self._clothes_labels.append('gloves')
        self._clothes_colors = {
            name: [np.random.randint(0, 255) for _ in range(3)] 
            for name in self._clothes_labels}
        
        self.last_person_preds = []
        self.last_clothes_preds = []
        self.last_keypoints_preds = []          
        
        self.mem_const = 25
        self.kp_mem = self.mem_const
        self.clothes_mem = self.mem_const
        
    def draw_boxes(self, image, boxes, colors, thikness=3):
        """
        Метод отрисовки боксов.
        Args:
            image: np.array, изображение
            boxes: list of dict, информаци о боксах (координаты и лейблы)
            colors: dict, цвета для отрисовки
            thikness: int, толщина линий
        """
        img = image.copy()
        for box in boxes:
            plot_one_box([box['x1'], box['y1'], box['x2'], box['y2']],
                          img, colors[box['label']], box['label'], thikness)
        return img
            
    def draw_predicts(self, image, person_preds: List = [], clothes_preds: List = [],
             keypoints_preds: Dict = {}) -> np.ndarray:
        """
        Метод отрисовки боксов.
        Args:
            image: np.array, изображение
            person_preds: list of dict, предсказание персон детекшена
            clothes_preds: list of dict, предсказание одежды
            keypoints_preds: dict, предсказания ключевых точек
        """
        img = image.copy()

        if person_preds:
            self.last_person_preds = person_preds
        img = self.draw_boxes(img, self.last_person_preds, self._person_colors)
        
        if clothes_preds:
            self.clothes_mem = self.mem_const
            self.last_clothes_preds = clothes_preds
        
        if self.clothes_mem > 0:
            self.clothes_mem -= 1
            img = self.draw_boxes(img, self.last_clothes_preds, self._clothes_colors)
            
        if keypoints_preds:
            self.kp_mem = self.mem_const
            self.last_keypoints_preds = keypoints_preds
        
        if self.kp_mem > 0 and self.last_keypoints_preds:
            self.kp_mem -= 1
            draw_joints(img, self.last_keypoints_preds, LINK_PAIRS, COLORS)
            draw_points(img, self.last_keypoints_preds, POINT_COLOR, False, 5)
        
        return img

class ParallelDetector:
    """
    Чисто формальный класс для метода, который вызывается фоновым потоков
    """
    @staticmethod
    def run_predicts(cfg, q_image, q_predict):
        """
        Метод в котором происходят все предсказания. Работает параллельно с основным потоком.
        Args:
            cfg: dict, Конфигурационный файл для моделей
            q_image: multiprocessing.Queue, для передачи входных данных из главного потока
            q_predict: multiprocessing.Queue, для передачи предсказаний в главный поток
        """
        detector = Detector(cfg)
        while True:
            if q_image.empty():
                continue
            item = q_image.get_nowait()        
            if item['exit']:
                break
            start_stages = time.time()
            (person_preds, clothes_preds, keypoints_preds, labels, meta) = detector.stages_flow(item['image'], item['flip'])
            end_stages = time.time()
            stages_time = max(end_stages - start_stages, 0.000000001)
            meta['stages_time'] = stages_time
            q_predict.put_nowait({'person_preds': person_preds, 
                                  'clothes_preds': clothes_preds,
                                  'keypoints_preds': keypoints_preds,
                                  'labels': labels,
                                  'meta': meta})
        detector.clothes_model.empty_cuda_memory()
        detector.person_model.empty_cuda_memory()
        detector.keypoints_model.empty_cuda_memory()
        q_predict.close()
        q_image.close()
        
        
class WearingDetector:
    """
    Класс с загрузчиком данных. В данном классе происходят все основные действия -
    отправляются данные в модель и отрисовываются предсказания.
    """
    def __init__(self, cfg: dict, pipe: (int, str) = None):
        self._config = cfg
        self.common_cfg = self._config['VIDEOSTREAM']

        if pipe is None:
            pipe = self.common_cfg['PIPE']
        self.set_loader(
            pipe, self._config['CLOTHES_MODEL']['IMG_SIZE'])
            
        self.artist = Artist(person_detection_classes, copy.deepcopy(wear_detection_classes))
        
        self.q_image = mp.Queue()
        self.q_predict = mp.Queue()
        self.p_process = mp.Process(target=ParallelDetector.run_predicts, args=(cfg, self.q_image, self.q_predict))
        self.p_process.daemon = True
        self.p_process.start()
                
    def set_loader(self, pipe: (int, str), img_size: int = None):
        """
        Инициализирует загрузчик кадров.

        Args:
            pipe: адрес откуда берется видеопоток.
            img_size: новый размер для входных кадров.
        """
        if img_size is None:
            img_size = self._config['CLOTHES_MODEL']['IMG_SIZE']
        if hasattr(self, 'loader'):
            self.release()
        self.loader = LoadWebcam(pipe, img_size)
        logger.debug(f'set_loader from pipe: {self.common_cfg["PIPE"]}')
    
    def detect(self):
        """
        Генератор, который содержит полный цикл детекции и обрабатывает
        кадры из видеопотока.
        """
        logger.info('run detect')
        t0 = time.time()

        for _, img0, flip in self.loader:
            if self.q_image.empty():
                self.q_image.put_nowait({'image': img0, 'flip':flip, 'exit': False})
            if self.q_predict.empty():
                img0 = self.artist.draw_predicts(img0, [], [], [])
                yield img0, []
                continue
            else:
                item = self.q_predict.get_nowait()
                draw_img = img0.copy()

                times = item['meta']
                fps = 1 / times['stages_time']

                if self.common_cfg['DRAW_LABELS']:
                    draw_img = self.artist.draw_predicts(
                        draw_img,
                        item['person_preds'],
                        item['clothes_preds'],
                        item['keypoints_preds'])

                if self.common_cfg['LOG_DETECTION_TIME']:
                    for stage_name, detect_time in times.items():
                        logger.info(f'{stage_name} = {detect_time}')
                    logger.info(f'fps = {fps}')

                if self.common_cfg['DRAW_TEXT']:
                    cv2.putText(draw_img, f'{round(times["stages_time"], 5)} sec', (7, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(draw_img, f'{round(fps,2)} fps', (7, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                yield draw_img, item['labels']

        logger.info(
            f'stop detect,detection time = {(time.time() - t0):.3} sec')
            
    def release(self):
        """Освобождение видеопотока."""
        self.q_image.put_nowait({'exit': True})
        if self.loader.cap.isOpened():
            self.loader.cap.release()
            logger.debug(f'release capture')


if __name__ == '__main__':
    import sys
    sys.path.append('./yolov5')
