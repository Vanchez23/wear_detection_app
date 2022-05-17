from pathlib import Path
import yaml
from wearing_detector.hrnet_model import HRNetModel

def test_hrnet_model():
    cfg_path = Path('wearing_detector/configs/wearing_detector_config.yaml')
    imgs_path = Path('tests/wearing_detector/imgs')

    assert cfg_path.exists(), f'config file {cfg_path} doesn\'t exists'
    assert imgs_path.exists(), f'samples path {imgs_path} doesn\'t exists'

    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = cfg['KEYPOINTS_MODEL']
    model = HRNetModel(cfg)
    predicts = {}
    for format_ in ('*.jpg','*.JPG'):
        for img_path in imgs_path.rglob(format_):
            predicts[img_path] = model(str(img_path))

    assert predicts