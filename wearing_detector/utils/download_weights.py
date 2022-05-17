import os
import os.path as osp
import yaml
from loguru import logger
from pathlib import Path
from wearing_detector.utils.google_utils import gdrive_download as gdd


def google_drive(conf):
    gdd(conf['id'], conf['path'])


def curl(conf):
    os.system(f"curl -L {conf['url']} -o {conf['path']}")


def main(config_name='download_config.yaml', overwrite=False):
    with open(config_name) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for model_type in config.values():
        try:
            if not overwrite and osp.exists(model_type['path']):
                raise FileExistsError(
                    f"File {model_type['path']} already exists. Set overwrite=True to overwrite file")
            os.makedirs(osp.dirname(model_type['path']), exist_ok=True)

            eval(model_type['type'])(model_type)
        except FileExistsError as fex:
            logger.warning(fex)
        except Exception as ex:
            logger.error(ex)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_name',
        default=Path('wearing_detector/configs/download_config.yaml'),
        type=str,
        help='path to download config file')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='overwrite existing files')
    opt = parser.parse_args()
    main(opt.config_name, opt.overwrite)
