import math
from typing import List, Tuple, Dict

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from .hrnet import get_pose_net
from wearing_detector.utils.torch_utils import select_device
from wearing_detector.utils.utils_hrnet import get_max_preds


class HRNetModel:

    def __init__(self, cfg, make_preprocess=True):
        """

        Args:
            cfg: конфиг для всей модели.
            make_preprocess: выполнить предобработку данных перед подачей в модель.
        """

        # cudnn related setting
        cudnn.benchmark = cfg['CUDNN']['BENCHMARK']
        torch.backends.cudnn.deterministic = cfg['CUDNN']['DETERMINISTIC']
        torch.backends.cudnn.enabled = cfg['CUDNN']['ENABLED']

        self.cfg = cfg
        self.device = select_device(self.cfg['DEVICE'])
        self.img_size = list(self.cfg['IMG_SIZE'])
        self.make_preprocess = make_preprocess
        self.keypoints_names = self.cfg['KEYPOINTS_NAMES']
        self.model = get_pose_net(self.cfg, is_train=False)
        self.model.load_state_dict(
            torch.load(
                self.cfg['WEIGHTS'],
                map_location=self.device),
            strict=False)
        if self.device.type != 'cpu':
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.cfg['GPUS']).cuda()
        self.model.eval()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def _get_dir(self, src_point: (np.ndarray, List), rot_rad: int = 0):
        """
        Вычислить координаты точки с учетом вращения.

        Args:
            src_point: исходная точка.
            rot_rad: радианы вращения.

        Returns:
            Точка с учетом вращения.
        """
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def _get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:

        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def _get_affine_transform(self, center: np.ndarray,
                              scale: (np.ndarray, List),
                              output_size: (np.ndarray, List),
                              shift: np.ndarray = np.array([0, 0], dtype=np.float32),
                              inv: bool = False):
        """
        Вычислить матрицу аффинных преобразований.

        Args:
            center: центр bounding box оригинального изображения.
            scale: масштаб bounding box по отношению к изображению.
            output_size: размер изображения к которому преобразовывать изображение.
            shift: сдвиг изображения.
            inv: флаг для вычисления обратной матрицы трансформации.

        Returns:
            Матрица аффинных преобразований.
        """
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        src_dir = self._get_dir([0, src_w * -0.5])
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def _affine_transform(self, point: (np.ndarray, List), transform: np.ndarray) -> np.ndarray:
        """
        Применяет матрицу преобразований.

        Args:
            point: пара координат для трансформации.
            transform: матрица трансформации.

        Returns:
            Преобразованная пара координат.
        """
        new_point = np.array([point[0], point[1], 1.]).T
        new_point = np.dot(transform, new_point)
        return new_point[:2]

    def _get_center(self, bbox: List) -> np.ndarray:
        """
        Вычисляет центр bounding box.

        Args:
            bbox: координаты bounding box в формате (x1,y1,x2,y2).

        Returns:
            Центр bounding box.
        """
        x, y, w, h = bbox
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        return center

    def _get_scale(self, bbox_w: int, bbox_h: int, width: int, height: int) -> np.ndarray:
        """
        Вычисляет масштаб bounding box из размеров bounding box и размеров изображения.

        Args:
            bbox_w: ширина bounding box.
            bbox_h: высота bounding box.
            width: ширина изображения.
            height: высота изображения.

        Returns:
            Масштаб bounding box.
        """
        aspect_ratio = height / width
        if bbox_w > aspect_ratio * bbox_h:
            bbox_h = bbox_w / aspect_ratio
        elif bbox_w < aspect_ratio * bbox_h:
            bbox_w = bbox_h * aspect_ratio

        scale = np.array([bbox_w, bbox_h], dtype=np.float32) * 1.25
        return scale

    def transform_preds(self, coords: np.ndarray, center: Tuple[int, int],
                        scale: (np.ndarray, List), output_size: List) -> np.ndarray:
        """
        Преобразует предсказанные координаты к оригинальному изображению(до обработки).

        Args:
            coords: предсказанные координаты.
            center: центр bounding box оригинального изображения.
            scale: масштаб изображения.
            output_size: размер изображения к которому преобразовывать предсказания.

        Returns:
            Преобразованные координаты.
        """
        target_coords = np.zeros(coords.shape)
        trans = self._get_affine_transform(
            center, scale, output_size, inv=True)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = self._affine_transform(
                coords[p, 0:2], trans)
        return target_coords

    def preprocess_img(self, img: np.ndarray, bbox: List) -> Tuple[np.ndarray, Dict]:
        """
        Предобработка изображения (сдвиг, масштабирование и вырез bounding box).

        Args:
            img: оригинальное изображение.
            bbox: координаты bounding box в формате (x1,y1,x2,y2).

        Returns:
            Обработанное изображение и информация о преобразованиях (для обратных операций).
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        center = self._get_center(bbox)
        scale = self._get_scale(
            bbox[2],
            bbox[3],
            self.img_size[1],
            self.img_size[0])

        trans = self._get_affine_transform(
            center, scale, (self.img_size[1], self.img_size[0]))
        cropped_img = cv2.warpAffine(
            img, trans, (self.img_size[1], self.img_size[0]), flags=cv2.INTER_LINEAR)

        meta = {'center': center,
                'scale': scale,
                'rotation': 0,
                'trans': trans}

        return cropped_img, meta

    def preprocess_bbox(self, bbox: List, width: int,
                        height: int) -> np.ndarray:
        """
        Корректировка координат bounding box с учетом границ изображения.

        Args:
            bbox: координаты bounding box.
            width: ширина изображения.
            height: высота изображения.

        Returns:
            Подкорректированные координаты.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x1 = np.max((0, x1))
        y1 = np.max((0, y1))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))

        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)

    def get_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Получить тензор из изображения."""
        img_tensor = self.transforms(img)
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def predict_tensor(self, tensor: torch.Tensor) -> torch.Tensor:

        return self.model(tensor)

    def postprocess(self, heatmaps: np.ndarray, meta: Dict) -> Tuple[np.ndarray, List]:
        """
        Обработка и отбор предсказаний модели.

        Args:
            heatmaps: тепловые карты предсказанные моделью. Размерность [batch_size, num_joints, height, width].
            meta: дополнительная информация об исходном изображении.

        Returns:
            Преобразованные предсказания и максимальные значения по тепловым картам.
        """
        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        coords, maxvals = get_max_preds(heatmaps)

        for p in range(coords.shape[1]):
            hm = heatmaps[0][p]
            px = int(math.floor(coords[0][p][0] + 0.5))
            py = int(math.floor(coords[0][p][1] + 0.5))
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array(
                    [
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px]
                    ]
                )
                coords[0][p] += np.sign(diff) * .25

        new_preds = coords.copy()

        # Transform back
        new_preds[0] = self.transform_preds(
            coords[0], meta['center'], meta['scale'], [
                heatmap_width, heatmap_height]
        )

        return new_preds[0], maxvals[0]

    def predict(self, img: (str, np.ndarray), bbox: List = None, meta: Dict = None) -> Tuple:
        """
        Выполнение всего пайплайна для предсказания.

        Args:
            img: входное изображение.
            bbox: координаты bounding box.
            meta: дополнительная информация об изображении, вроде шасштаба и центра для
                  обратного преобразования изображения(для того случая, если входное изображение было
                  предобработано раньше).

        Returns:
            Обработанные предсказания.
        """
        return self(img, bbox, meta)

    def __call__(self, img: (str, np.ndarray), bbox: List = None, meta: Dict = None) -> Tuple:
        """
        Выполнение всего пайплайна для предсказания.

        Args:
            img: входное изображение.
            bbox: координаты bounding box.
            meta: дополнительная информация об изображении, вроде шасштаба и центра для
                  обратного преобразования изображения(для того случая, если входное изображение было
                  предобработано раньше).

        Returns:
            Обработанные предсказания.
        """
        if isinstance(img, str):
            img = cv2.imread(img)

        if self.make_preprocess:
            if bbox is None:
                bbox = [0, 0, img.shape[1], img.shape[0]]
            bbox = self.preprocess_bbox(bbox, img.shape[1], img.shape[0])
            img, meta = self.preprocess_img(img, bbox)
        with torch.no_grad():
            tensor = self.get_tensor(img)
            output = self.predict_tensor(tensor)
            coords, confs = self.postprocess(
                output.clone().cpu().numpy(), meta)

        return coords, confs

    def predict_bboxes(self, image: np.ndarray, bboxes: List,
                       ndigits: int = 3) -> List:
        """
        Сделать предсказания для всех bounding box.

        Args:
            image: входное изображение.
            bboxes: набор bounding box.
            ndigits: округление вероятностей.

        Returns:
            Предсказания для bounding box.
        """
        answers = []
        for bbox in bboxes:
            result = dict()
            coords, confs = self(image, bbox)
            for i, k_name in enumerate(self.keypoints_names):
                result[k_name] = {'x': float(coords[i][0]),
                                  'y': float(coords[i][1]),
                                  'proba': float(round(confs[i][0], ndigits))}
            answers.append(result)

        return answers

    def empty_cuda_memory(self):
        "Очищает кэш видеопамяти"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()