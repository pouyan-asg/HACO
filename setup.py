import sys
from setuptools import setup, find_packages

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name="haco",
    packages=find_packages(include=["haco*"]),
    install_requires=[
        "yapf==0.30.0",
        "tensorflow==2.3.1",
        "tensorflow-probability==0.11.1",
        "tensorboardX",
        "metadrive-simulator==0.2.4",
        "imageio",
        "easydict",
        "pyyaml",
        "gym==0.19.0",
        "ray==1.0.0",
        "stable_baselines3",
        "ephem",
        "h5py",
        "imgaug",
        "lmdb",
        "loguru==0.3.0",
        "networkx",
        "pandas",
        "py-trees==0.8.3",
        "pygame==1.9.6",
        "scikit-image",
        "shapely",
        "terminaltables",
        "tqdm",
        "xmlschema",
    ],
)

