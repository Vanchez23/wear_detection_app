import numpy as np
import cv2

# Xiaochu Style
# (R,G,B)
COLORS = [(179, 0, 0), (228, 26, 28), (255, 255, 51),
          (49, 163, 84), (0, 109, 45), (255, 255, 51),
          (240, 2, 127), (240, 2, 127), (240, 2,
                                         127), (240, 2, 127), (240, 2, 127),
          (217, 95, 14), (254, 153, 41), (255, 255, 51),
          (44, 127, 184), (0, 0, 255)]

LINK_PAIRS = [
    [15, 13], [13, 11], [11, 5],
    [12, 14], [14, 16], [12, 6],
    [3, 1], [1, 2], [1, 0], [0, 2], [2, 4],
    [9, 7], [7, 5], [5, 6],
    [6, 8], [8, 10],
]

POINT_COLOR = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
               (240, 2, 127), (240, 2, 127),
               (255, 255, 51), (255, 255, 51),
               (254, 153, 41), (44, 127, 184),
               (217, 95, 14), (0, 0, 255),
               (255, 255, 51), (255, 255, 51), (228, 26, 28),
               (49, 163, 84), (179, 0, 0), (0, 109, 45),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142)]

KEYPOINTS_NAMES = ['nose',
                   'l_eye',
                   'r_eye',
                   'l_ear',
                   'r_ear',
                   'l_shoulder',
                   'r_shoulder',
                   'l_elbow',
                   'r_elbow',
                   'l_wrist',
                   'r_wrist',
                   'l_hip',
                   'r_hip',
                   'l_knee',
                   'r_knee',
                   'l_ankle',
                   'r_ankle']


class ColorStyle:
    def __init__(self, color, link_pairs, point_color, keypoints_names):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.color)):
            self.link_pairs[i].append(tuple(np.array(self.color[i]) / 255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i]) / 255.))

        self.keypoints_names = keypoints_names


class Args(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

# def draw_joints(img, joints, link_pairs, scores, keypoints_names, joint_colors, point_colors,
#                 joint_thres = 0.2, copy_img=True):
#     if copy_img:
#         img = img.copy()
#     for k, link_pair in enumerate(link_pairs):
#         if scores[link_pair[0],0] < joint_thres \
#             or scores[link_pair[1],0] < joint_thres:
#             continue
#         pt1 = tuple(joints[link_pair[0]])
#         pt2 = tuple(joints[link_pair[1]])
#         cv2.line(img, pt1, pt2, list(joint_colors[k])[::-1], 7, )
#     for i, joint in enumerate(joints):
#         if scores[link_pair[0],0] < joint_thres \
#             or scores[link_pair[1],0] < joint_thres:
#             continue
#         point = tuple(joint)
#         cv2.circle(img, point, 20, (0,0,0), -1)
#         cv2.circle(img, point, 15, list(point_colors[i])[::-1], -1)
#         point = (point[0]+20,point[1])
#         cv2.putText(img, keypoints_names[i], point, cv2.FONT_HERSHEY_SIMPLEX, 1, list(point_colors[i])[::-1],2)
#
#     if copy_img:
#         return img


def draw_joints(img, keypoints, link_pairs, joint_colors, joint_thres=0.2):
    for k, link_pair in enumerate(link_pairs):
        if (keypoints[link_pair[0]]['conf'] < joint_thres
                or keypoints[link_pair[1]]['conf'] < joint_thres):
            continue
        pt1 = keypoints[link_pair[0]]['point']
        pt2 = keypoints[link_pair[1]]['point']
        cv2.line(img, pt1, pt2, list(joint_colors[k])[::-1], 7)


def draw_points(img, keypoints, point_colors,
                put_text=True, radius=15, kp_thres=0.5):
    for idx, values in keypoints.items():
        if values['conf'] < kp_thres:
            continue
        point = values['point']
        color = list(point_colors[idx])[::-1]
        cv2.circle(img, point, radius + 5, 0, -1)
        cv2.circle(img, point, radius, color, -1)
        if put_text:
            cv2.putText(
                img,
                values['label'],
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                0,
                3)
            cv2.putText(
                img,
                values['label'],
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2)


XIAOCHU_STYLE = ColorStyle(COLORS, LINK_PAIRS, POINT_COLOR, KEYPOINTS_NAMES)
# xiaochu_style = ColorStyle(color1, link_pairs1, point_color1, keypoints_names)
