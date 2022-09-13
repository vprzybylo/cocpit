# read the contents of README file
from os import path

from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("requirements_dev.txt") as f:
    required = f.read().splitlines()

setup(
    # Name of the package
    name="cocpit",
    packages=find_packages("."),
    version="2.0.0",
    description="Classification of cloud particle imagery and thermodynamics",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    author="Vanessa Przybylo",
    author_email="vprzybylo@albany.edu",
    # Either the link to your github or to your website
    url="https://github.com/vprzybylo/cocpit",
    # Link from which the project can be downloaded
    download_url="https://github.com/vprzybylo/cocpit.git",
    python_requires=">=3.7",
    # List of packages to install
    # parse_requirements() returns generator of pip.req.InstallRequirement objects
    install_reqs=required,
    include_package_data=True,
    # https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
    ],
)
