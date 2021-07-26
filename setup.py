import os

from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="cocpit",
    version="0.0.0",
    author="Vanessa Przybylo",
    author_email="vprzybylo@albany.edu",
    description=("Package for classifying ice crystal images from the CPI probe"),
    license="MIT",
    keywords="ice microphysics crystals cloud images classification convolution",
    url="https://vprzybylo.github.io/cocpit/",
    packages=["cocpit"],
    long_description=read("README.md"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: GPU :: NVIDIA CUDA :: 10.0",
        "Framework :: Jupyter",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    install_requires=[
        "certifi==2020.6.20",
        "cycler==0.10.0",
        "imutils==0.5.3",
        "joblib==0.16.0; python_version >= '3.6'",
        "kiwisolver==1.2.0; python_version >= '3.6'",
        "matplotlib==3.3.1",
        "numpy==1.19.1",
        "opencv-python==4.4.0.42",
        "pandas==1.1.1",
        "pickle5==0.0.11",
        "pillow==7.2.0",
        "pyparsing==2.4.7; python_version >= '2.6' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "pytz==2020.1",
        "scikit-learn==0.23.2; python_version >= '3.6'",
        "scipy==1.5.2; python_version >= '3.6'",
        "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "sklearn==0.0",
        "threadpoolctl==2.1.0; python_version >= '3.5'",
        "torch==1.4.0",
        "torchvision==0.5.0",
    ],
)
