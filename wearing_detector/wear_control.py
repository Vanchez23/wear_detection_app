import numpy as np
from typing import Dict, List, Tuple
import pandas as pd


class Skeleton:

    def __init__(self, keypoints: (Dict, pd.Series)):
        """

        Args:
            keypoints: названия ключевых точек и их координаты.
        """

        if isinstance(keypoints, pd.Series):
            self.keypoints = {
                name: eval(str_tup) for name,
                str_tup in keypoints.items()}
        else:
            self.keypoints = {value['label']: value['point']
                              for _, value in keypoints.items()}

        self.head = {
            name: self.keypoints[name] for name in [
                'l_ear',
                'l_eye',
                'nose',
                'r_eye',
                'r_ear']}
        self.hands = {
            'left': {name.split('_')[1]: self.keypoints[name] for name in ['l_shoulder', 'l_elbow', 'l_wrist']},
            'right': {name.split('_')[1]: self.keypoints[name] for name in ['r_shoulder', 'r_elbow', 'r_wrist']}}
        self.torso = {
            name: self.keypoints[name] for name in [
                'l_shoulder',
                'l_hip',
                'r_shoulder',
                'r_hip']}
        self.legs = {name: self.keypoints[name] for name in
                     ['l_hip', 'l_knee', 'l_ankle', 'r_hip', 'r_knee', 'r_ankle']}

    def _get_distances(self, part: Dict,
                       center_point: (List, np.ndarray)):
        """
        Получить евклидово расстояние между точками части тела и заданной точкой.

        Args:
            part: словарь с ключевыми точками части тела.
            center_point: точка до которой получить расстояния.

        Returns:
            Расстояния от точек части тела до заданной точки.
        """
        dists = dict()
        for name, point in part.items():
            dists[name] = np.sqrt(np.power(np.abs(part[name][0] - center_point[0]), 2)
                                  + np.power(np.abs(part[name][1] - center_point[1]), 2))

        return dists

    def _part_inside(self, part: Dict, bbox: (
            List, np.ndarray), margin: float = 0.) -> Tuple[float, Dict]:
        """
        Определить находится ли часть тела внутри bounding box.

        Args:
            part: словарь с ключевыми точками части тела.
            bbox: координаты bbox в формате [x1, y1, x2, y2].
            margin: отступы по периметру bounding box.

        Returns:
            Вероятность, что часть тела находится внутри bounding box и точки, которые находятся внутри.
        """
        if isinstance(margin, (int, float)):
            margin = [margin] * len(bbox)
        left, top, right, bottom = bbox
        points_inside = {name: 0 for name in part.keys()}
        for name, point in part.items():
            if (left - margin[0] <= point[0] <= right + margin[1] and
                    top - margin[2] <= point[1] <= bottom + margin[3]):
                points_inside[name] = 1

        points_inside_list = list(points_inside.values())
        return sum(points_inside_list) / len(points_inside_list), points_inside

    def head_near(self, bbox: (List, np.ndarray),
                  eye_nose_dist_coef: float = 2., repeat_eye_nose_dist_coef: int = 2, min_eye_nose_dist: float = 0.,
                  ears_dist_coef: float = 1, repeat_ears_dist_coef: int = 2, min_ears_dist: float = 0.,
                  repeat_dist_coef: int = 2, part_bbox_dist: float = 10.) -> Tuple[float, Tuple]:
        """
        Определить находится ли голова около bounding box.


        Args:
            bbox: координаты bbox в формате [x1, y1, x2, y2].
            eye_nose_dist_coef: коэффициент для изменения расстояния между глазом и носом.
            repeat_eye_nose_dist_coef: сколько раз повторить результирующее расстояние между глазом и носом.
            min_eye_nose_dist: минимальное расстояние между глазом и носом.
            ears_dist_coef: коэффициент для изменения расстояния между ушами.
            repeat_ears_dist_coef: сколько раз повторить результирующее расстояние между ушами.
            min_ears_dist: минимальное расстояние между ушами.
            repeat_dist_coef: сколько раз повторить расстояние по-умолчанию.
            part_bbox_dist: значение расстояния по-умолчанию между головой и bounding box.

        Returns:
            Вероятность, что часть тела находится внутри bounding box и расстояние между ключевыми точками.
        """
        center_bbox = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        dists = self._get_distances(self.head, center_bbox)
        dist_median = np.median(np.array(list(dists.values())))

        eyes_dist = None
        eye_nose_dist = None
        ears_dist = self._get_distances(
            {'l_ear': self.head['l_ear']}, self.head['r_ear'])['l_ear']

        if ears_dist > min_ears_dist:
            part_bbox_dist = ears_dist * ears_dist_coef
            max_dist = part_bbox_dist * repeat_ears_dist_coef
        else:
            eyes_nose_dist = self._get_distances({'l_eye': self.head['l_eye'], 'r_eye': self.head['r_eye']},
                                                 self.head['nose'])
            eye_nose_dist = max(
                eyes_nose_dist['l_eye'],
                eyes_nose_dist['r_eye'])
            if eye_nose_dist > min_eye_nose_dist:
                part_bbox_dist = eye_nose_dist * eye_nose_dist_coef
                max_dist = part_bbox_dist * repeat_eye_nose_dist_coef
            else:
                max_dist = part_bbox_dist * repeat_dist_coef

        score = int(dist_median <= max_dist)
        if score:
            score = 1 - np.abs(dist_median - part_bbox_dist) / max_dist

        return score, (dist_median, eyes_dist, eye_nose_dist,
                       ears_dist, part_bbox_dist, max_dist)

    def head_inside(self, bbox: (List, np.ndarray), margin: float = 0) -> Tuple[float, Dict]:
        """
        Определить находится ли голова внутри bounding box.

        Args:
            bbox: координаты bbox в формате [x1, y1, x2, y2].
            margin: отступы по периметру bounding box.

        Returns:
            Вероятность, что голова находится внутри bounding box и точки, которые находятся внутри.
        """
        return self._part_inside(self.head, bbox, margin)

    def hand_inside(self, bbox: (List, np.ndarray),
                    margin: float = 0, side: str = 'left') -> Tuple[float, Dict]:
        """
        Определить находится ли рука внутри bounding box.

        Args:
            bbox: координаты bbox в формате [x1, y1, x2, y2].
            margin: отступы по периметру bounding box.

        Returns:
            Вероятность, что рука находится внутри bounding box и точки, которые находятся внутри.
        """
        if side.lower() not in ('left', 'right'):
            raise ValueError(
                f"param 'side' should be 'left' or 'right', got: {side}")
        return self._part_inside(self.hands[side], bbox, margin)

    def wrist_inside(self, bbox: (List, np.ndarray),
                     margin: float = 0, side: str = 'left') -> Tuple[float, Dict]:
        """
        Определить находится ли кисть внутри bounding box.

        Args:
            bbox: координаты bbox в формате [x1, y1, x2, y2].
            margin: отступы по периметру bounding box.

        Returns:
            Вероятность, что кисть находится внутри bounding box и точки, которые находятся внутри.
        """
        if side.lower() not in ('left', 'right'):
            raise ValueError(
                f"param 'side' should be 'left' or 'right', got: {side}")

        part = {'wrist': self.hands[side]['wrist']}
        return self._part_inside(part, bbox, margin)

    def torso_inside(self, bbox: (List, np.ndarray), margin: float = 0) -> Tuple[float, Dict]:
        """
        Определить находится ли туловище внутри bounding box.

        Args:
            bbox: координаты bbox в формате [x1, y1, x2, y2].
            margin: отступы по периметру bounding box.

        Returns:
            Вероятность, что туловище находится внутри bounding box и точки, которые находятся внутри.
        """
        return self._part_inside(self.torso, bbox, margin)

    def legs_inside(self, bbox: (List, np.ndarray), margin: float = 0) -> Tuple[float, Dict]:
        """
        Определить находятся ли ноги внутри bounding box.

        Args:
            bbox: координаты bbox в формате [x1, y1, x2, y2].
            margin: отступы по периметру bounding box.

        Returns:
            Вероятность, что ноги находятся внутри bounding box и точки, которые находятся внутри.
        """
        return self._part_inside(self.legs, bbox, margin)

    def wrist_near(self, bbox: (List, np.ndarray), side: str = 'left',
                   max_dist: float = 200., part_bbox_side: int = 3) -> Tuple[float, float]:
        """
        Определить находится ли кисть рядом с bounding box.

        Args:
            bbox: координаты bbox в формате [x1, y1, x2, y2]

        Returns:
            Вероятность, что кисть находится рядом и расстояние кисти до bounding box.
        """
        if side.lower() not in ('left', 'right'):
            raise ValueError(
                f"param 'side' should be 'left' or 'right', got: {side}")

        hand = self.hands[side.lower()]
        center_bbox = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        max_bbox_side = max(
            np.abs(
                bbox[0] -
                bbox[2]),
            np.abs(
                bbox[1] -
                bbox[3]))

        dist_wrist_bbox = np.sqrt(np.abs(hand['wrist'][0] - center_bbox[0]) ** 2
                                  + np.abs(hand['wrist'][1] - center_bbox[1]) ** 2)

        score_bin = dist_wrist_bbox <= max_dist
        if score_bin:
            score = 1 - np.abs(dist_wrist_bbox -
                               max_bbox_side / part_bbox_side) / max_dist
        else:
            score = int(score_bin)

        return score, dist_wrist_bbox
