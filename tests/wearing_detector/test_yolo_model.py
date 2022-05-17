import sys
sys.path.append('wearing_detector/yolov5')
from pathlib import Path
import numpy as np
import yaml
from wearing_detector.yolo_model import YoloModel
from wearing_detector.constants import wear_detection_classes
print('wear_detection_classes = ', wear_detection_classes)
from wearing_detector.utils.general import xywh2xyxy

def get_labels(imgs_path: Path):
    labels = {}
    for format_ in ('*.jpg','*.JPG'):
        for path in imgs_path.rglob(format_):
            label_name = path.parent/(path.stem+'.txt')
            assert label_name.exists(), f'label {label_name} doesn\'t exists'

            with open(label_name) as f:
                file_labels = []
                for line in f:
                    line = line.strip().split(' ')
                    line = [float(i) for i in line]
                    line[0] = int(line[0])
                    xywh2xyxy(np.array([line[1:]]))
                    file_labels.append(line)
            labels[path] = file_labels
    return labels

def predicts2format(preds, classes):
    new_preds = {}
    for img_name, preds_img in preds.items():
        cur_preds = np.zeros((len(preds_img), 6))
        for i,pred in enumerate(preds_img):
            cur_preds[i][0] = classes.index(pred['label'])
            cur_preds[i][1] = pred['x1']
            cur_preds[i][2] = pred['y1']
            cur_preds[i][3] = pred['x2']
            cur_preds[i][4] = pred['y2']
            cur_preds[i][5] = pred['conf']
        new_preds[img_name] = cur_preds
    return new_preds

def test_yolo_model():
    cfg_path = Path('wearing_detector/configs/wearing_detector_config.yaml')
    imgs_path = Path('tests/wearing_detector/imgs')
    assert cfg_path.exists(), f'config file {cfg_path} doesn\'t exists'
    assert imgs_path.exists(), f'samples path {imgs_path} doesn\'t exists'

    print(wear_detection_classes)

    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = cfg['CLOTHES_MODEL']
    model = YoloModel(cfg['WEIGHTS'],
                      cfg['DEVICE'],
                      cfg['IMG_SIZE'],
                      cfg['CONF_THRES'],
                      cfg['IOU_THRES'],
                      cfg['AGNOSTIC_NMS'],
                      cfg['PADDING'],
                      make_preprocess=True)
    classes = model.classes
    sorted(classes)
    sorted(wear_detection_classes)
    assert classes == wear_detection_classes, 'classes are not the same'

    # labels = get_labels(imgs_path)
    predicts = {}
    for format_ in ('*.jpg','*.JPG'):
        for img_path in imgs_path.rglob(format_):
            predicts[img_path] = model(str(img_path),normalize_output=True)
    # predicts = predicts2format(predicts, model.classes)

    assert predicts
