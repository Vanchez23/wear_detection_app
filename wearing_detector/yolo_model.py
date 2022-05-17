import os

import numpy as np
import argparse
import torch
import cv2
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple

from wearing_detector.utils.general import non_max_suppression, scale_coords
from wearing_detector.utils.torch_utils import select_device
from wearing_detector.utils.datasets import LoadImages, letterbox


class YoloModel:

    def __init__(self, weights: str, device: str = '0',
                 img_size: int = 416, conf_thres: float = 0.3, iou_thres: float = 0.6, agnostic: bool = True,
                 padding: float = 0, make_preprocess: bool = True):
        """

        Args:
            weights: путь под весов модели.
            device: устройство для вычислений. Номер видеокарты('0','1' и т.д.) или процессор('cpu').
            img_size: новый размер стороны изображения.
            conf_thres: минимальный порог уверенности классификации.
            iou_thres: максимальный порог для пересечения пары bounding box, при котором один из них отбрасывается.
            agnostic:
            padding: внешний отступ по всему периметру bounging box.
            make_preprocess: выполнить предобработку данных перед подачей в модель.
        """
        self.device = select_device(device)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic = agnostic
        self.make_preprocess = make_preprocess
        self.model = torch.load(weights, map_location=self.device)[
            'model'].float().fuse().eval()
        self.classes = self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(
            3)] for _ in range(len(self.classes))]
        self.padding = [padding] * 4 if isinstance(padding, int) else padding

        if self.device.type != 'cpu':
            self.model = torch.nn.DataParallel(
                self.model, device_ids=[int(device)]).cuda()

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        """
        Приводит изображение к нужному формату.

        Args:
            img: изображение для обработки.

        Returns:
            Отформатированное изображение.
        """
        img_res = letterbox(img, new_shape=self.img_size)[0]
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        img_res = img_res.transpose(2, 0, 1)

        return img_res

    def get_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Преобразовать изображение в подготовленный тензор.

        Args:
            img: входное изображение.

        Returns:
            Подготовленный тензор.
        """
        img_tensor = torch.from_numpy(img).to(self.device)
        img_tensor = img_tensor.float()  # uint8 to fp16/32
        img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def predict(self, tensor: torch.Tensor) -> List:

        with torch.no_grad():
            preds = self.model(tensor)

        return preds

    def postprocess(self, pred: (Tuple, List, np.ndarray), tensor_shape: (List, np.ndarray),
                    orig_shape: List, bbox: List, normalize_output=False) -> List:
        """
        Обработка предсказаний модели и отбор детекций.

        Args:
            pred: предсказания модели.
            tensor_shape: размер входного тензора.
            orig_shape: оригинальный размер изображения.
            bbox: координаты bounding box.

        Returns:
            Очищенные предсказания.
        """
        if isinstance(pred, (Tuple, List, np.ndarray)):
            pred = pred[0]

        pred = non_max_suppression(
            pred,
            self.conf_thres,
            self.iou_thres,
            agnostic=self.agnostic)

        new_preds = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    tensor_shape[2:], det[:, :4], orig_shape).round()
                det[:,
                    :4] += torch.tensor([bbox[0],
                                         bbox[1],
                                         bbox[0],
                                         bbox[1]]).to(self.device)
                for *xyxy, conf, cls in reversed(det):
                    for i in range(len(xyxy)):
                        xyxy[i] = xyxy[i].cpu().numpy()
                    if normalize_output:
                        xyxy[0] /= orig_shape[1]
                        xyxy[1] /= orig_shape[0]
                        xyxy[2] /= orig_shape[1]
                        xyxy[3] /= orig_shape[0]
                    new_preds.append({'conf': conf.to('cpu').numpy(),
                                      'label': self.classes[int(cls)],
                                      'x1': xyxy[0],
                                      'y1': xyxy[1],
                                      'x2': xyxy[2],
                                      'y2': xyxy[3]})

        return new_preds

    def __call__(self, img: (str, np.ndarray),
                 img_shape: List = None, bbox: List = None,
                 normalize_output=False) -> List:
        """
        Выполнение всего пайплайна для предсказания.

        Args:
            img: входное изображение.
            img_shape: размер изображения к которому нужно преобразовать.
            bbox: координаты bounding box.

        Returns:
            Обработанные предсказания.
        """
        if isinstance(img, str):
            img = cv2.imread(img)

        if bbox is None:
            bbox = [0, 0, img.shape[1], img.shape[0]]
        else:
            bbox = [max(0, bbox[0] - self.padding[0]),
                    max(0, bbox[1] - self.padding[1]),
                    min(bbox[2] + self.padding[2], img.shape[1]),
                    min(bbox[3] + self.padding[3], img.shape[0])]
        if self.make_preprocess:
            img = img[bbox[1]:bbox[3],
                      bbox[0]:bbox[2], :]
            img_shape = img.shape
            img = self.preprocess_img(img)

        with torch.no_grad():
            tensor = self.get_tensor(img)
            preds = self.predict(tensor)
            preds = self.postprocess(preds[0], tensor.shape, img_shape, bbox,
                                     normalize_output=normalize_output)

        return preds

    def empty_cuda_memory(self):
        "Очищает кэш видеопамяти"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def detect(opt) -> pd.DataFrame:
    img_loader = LoadImages(opt.imgs_path, opt.img_size)
    if opt.max_count is not None:
        img_loader.files = img_loader.files[:opt.max_count]
        img_loader.nf = len(img_loader.files)

    model = YoloModel(opt.weights, opt.device, opt.img_size,
                      opt.conf_thres, opt.iou_thres, opt.agnostic_nms,
                      make_preprocess=False)

    coords = []
    pbar = tqdm(img_loader)
    for p, img, img0, *_ in pbar:
        pbar.desc = f'{img_loader.count}/{img_loader.nf} {p}'
        preds = model(img, img0.shape)
        for pred in preds:
            pred['image_name'] = os.path.basename(p)
        coords.extend(preds)

    return pd.DataFrame(coords)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        type=str,
        help='pandas dataframe with clothes detects')
    parser.add_argument('--weights', type=str, help='path to model')
    parser.add_argument('--imgs_path', type=str, help='path to imgs')
    parser.add_argument(
        '--img-size',
        type=int,
        default=416,
        help='inference size (pixels)')
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.3,
        help='object confidence threshold')
    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.6,
        help='IOU threshold for NMS')
    parser.add_argument(
        '--device',
        default='0',
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '--agnostic_nms',
        action='store_true',
        help='class-agnostic NMS')
    parser.add_argument(
        '--max_count',
        type=int,
        action=None,
        help='max count imgs for predict')

    opt = parser.parse_args()

    with torch.no_grad():
        if not os.path.exists(os.path.dirname(opt.save_path)):
            raise ValueError(f'path {opt.save_path} not exists')
        if not opt.save_path.endswith('.csv'):
            opt.save_path += '.csv'

        df = detect(opt)

        df.to_csv(opt.save_path, index=False)
