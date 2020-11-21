[![Forks][forks-shield]][forks-url]
[![GitHub stars][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![PyPI download month][download-shield]][download-url]
[![GitHub release][release-shield]][release-url]

[download-shield]:https://img.shields.io/github/downloads/vprzybylo/cocpit/total?style=plastic
[download-url]: https://github.com/vprzybylo/cocpit/downloads
[release-shield]: https://img.shields.io/github/v/release/vprzybylo/cocpit?style=plastic
[release-url]:https://github.com/vprzybylo/cocpit/releases/
[forks-shield]: https://img.shields.io/github/forks/vprzybylo/cocpit?label=Fork&style=plastic
[forks-url]: https://github.com/vprzybylo/cocpit/network/members
[stars-shield]: https://img.shields.io/github/stars/vprzybylo/cocpit?style=plastic
[stars-url]: https://github.com/vprzybylo/cocpit/stargazers
[issues-shield]: https://img.shields.io/github/issues/vprzybylo/cocpit?style=plastic
[issues-url]: https://github.com/vprzybylo/cocpit/issues
[license-shield]: https://img.shields.io/github/license/vprzybylo/COCPIT?style=plastic
[license-url]: https://github.com/vprzybylo/cocpit/blob/master/LICENSE.md
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

<br />
<p align="center">
  <a>
    <img src="https://github.com/vprzybylo/cocpit/blob/master/logo.png" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">COCPIT</h3>
		    

  <p align="center">
    Classification of Cloud Particle Imagery and Thermodynamics 
    <br />
    <a href="https://vprzybylo.github.io/cocpit/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="filler">View Demo</a>
    ·
    <a href="https://github.com/vprzybylo/cocpit/issues">
    Report Bug</a>
    ·
    <a href="https://github.com/vprzybylo/cocpit/issues">Request Feature</a>
  </p>
</p>


# Table of Contents

* [About](#about)

  * [Built With](#built-with)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)

* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## About

A tool for the characterization of cloud particle images from the (<a href="http://www.specinc.com/cloud-particle-imager">CPI</a>) probe.  Three models are featured: one to separate cloud drops (liquid) from all other images, one to separate quality images of ice crystals from blurry, fragmented, and blank images, and the last to classify frozen hydrometeors from the remaining good images.  Probe images are classified into 11 categories (aggregates, blurry, budding rosettes, bullet rosettes, columns, compact irregulars, fragments, needles, plates, rimed aggregates and rimed columns) using a convolutional neural network.


### Preprocessing

* <b>SPHERES</b> model
  * Liquified cloud drops are filtered via a logistic regression model 
    * These predictors are used as independent variables to predict whether or not the particles are spherical (liquid):
      * variance of the laplacian (image blurriness)
      * height of image
      * width of image
      * number of contours
      * number of edges using the canny edge detector
      * standard deviation in the location of the edges 
      * contour area  (largest contour)
      * image contrast 
      * circularity (largest contour)
      * solidity (largest contour)
      * complexity  (largest contour)
      * equivalent diameter
      * convex perimeter
      * hull area  (largest contour)
      * perimeter  (largest contour)
      * aspect ratio (largest contour)
      * cutoff (percentage of pixels touching the border)
      * separation of the extreme points of the particle
      * filled circular area ratio
      * roundness (largest contour)
      * perim area ratio (largest contour)

* <b>SIFT</b> (Separate Ice for Training)
  * If the particles are predicted as not spherical, the SIFT model is then used to predict whether or not the image of ice is of good or bad quality' (e.g., blurry, fragmented or shattered, too dark, etc.)
  * The same predictors as the SPHERES model are used to filter bad images but a different training dataset was used.


### Built With

* <a href="https://www.python.org/"><a href="https://pytorch.org/docs/stable/torchvision/models.html">Python</a> </a> 
* <a href="https://pytorch.org/docs/stable/torchvision/models.html">Pytorch</a> 
* <a href="http://www.specinc.com/sites/default/files/software_and_manuals/CPI_Post Processing Software Manual_rev1.2_20120116.pdf">cpiview</a> 
  * Desktop-developed software will need to be used to extract ''sheets'' of CPI images from region of interest (ROI) files output from the CPI probe should new data be wished to be processed and classified.
* <a href="https://www.nvidia.com/en-us/">nvidia</a> 
  * Resources used: NVIDIA DGX-1 server utilizing Tesla V100 GPUs. This system is housed in the University at Albanys Tier-3 Data Center, and managed/maintained by the xCITE (ExTREME Collaboration, Innovation and TEchnology) laboratory. The base DGX-1 V100 system contains 8 Tesla V100 GPUs with a combined total of 40,960 CUDA (graphics) cores, 5120 Tensor cores, and 256GB of GPU memory, all linked by NVIDIAs 300GB/s NVLINK interconnect. The DGX-1 is optimized for data loading, data transformations, and training, which are all critical to the ML processes required by this project.

## Supported campaigns

|   Campaign   |              Aircraft               | Date              |
| :----------: | :---------------------------------: | ----------------- |
|   ARM IOP    | University of North Dakota Citation | March 2000        |
| CRYSTAL-FACE |            NASA's WB-57             | Jul 2002          |
| CRYSTAL-FACE | University of North Dakota Citation | Jul 2002          |
|   AIRS II    |           NRC Convair-580           | Nov 2003-Feb 2004 |
|    Midcix    |       NASA's WB-57 & Learjet        | Apr 2004-May 2004 |
|    Ice-L     |              NSF C-130              | Nov-Dec 2007      |
|    OLYPEX    | University of North Dakota Citation | Nov 2015-May 2016 |
|    MPACE     | University of North Dakota Citation | Sept-Oct 2004     |


# Installation

1. Clone the repo <br>
	git clone: [git@github.com:vprzybylo/cocpit.git](git@github.com:vprzybylo/cocpit.git)<br/>
## Usage 

Add video 

## Feature Descriptions

Include work flow diagram

## Prerequisites


## Roadmap

* Each image will have time-correlated in situ measurements added to the database via aircraft probes own through dierent cloud layers during field campaign initiatives. The CPI particle timestamp will be used to synchronize with other data systems on board to obtain environmental parameters, that include but are not limited to: location, atmospheric conditions, and other characteristics based on what was measured during the specic field program discussed.
* Provide functions to determine a relative likelihood for a particular thermodynamic property to make up a predefined sample space, such as habit type.
* It cannot be assumed that these particles initiated or grew within the environment at time of capture, hence, there is motivation to use trajectory analysis to track and associate particle characteristics with the environment in which they resided.

## License

Distributed under the MIT License.  See `LICENSE` for more information.

## Contact 

Vanessa Przybylo - vprzybylo@albany.edu


Project Link: [https://vprzybylo.github.io/cocpit/](https://vprzybylo.github.io/cocpit/)


## Acknowledgements
* V. Przybylo, K. Sulia, C. Schmitt, and Z. Lebo (collaborators on this project) would like to thank the Department of Energy for support under DOE Grant Number DE-SC0021033.
* Development support given by the Atmospheric Sciences Research Center ExTreme Collaboration, Innovation, and TEchnology (xCITE) Laboratory.


