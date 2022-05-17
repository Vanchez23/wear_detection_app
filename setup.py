#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name='wearing_detector',
    version='1.0',
    author='Ivan Bukun (OCRV)',
    url='unknown',
    description='Detecting specific clothing on a person',
    install_requires=[
        'pyrsistent==0.17.3',
        'PyYAML==5.3.1',
        'torch==1.8.0',
        'torchvision==0.9.0',
        'opencv-python==4.4.0.44',
        'numpy==1.19.2',
        'loguru==0.5.3',
        'pandas==1.1.5',
        'tqdm==4.54.1',
        'matplotlib==3.3.3',
        'scipy==1.5.4',
    ],
    packages=find_packages(exclude=('configs',)),
)