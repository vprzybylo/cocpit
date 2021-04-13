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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1L9afVqbOYwHh836MFwnEzy88l3tPycWz?authuser=1#scrollTo=MKD112KpRCOe)


<br />
<p align="center">
  <a>
    <img src="https://github.com/vprzybylo/cocpit/README_graphics/logo.png" alt="Logo" width="150" height="150">
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
  * [Installation](#installation)
  * [Prerequisites](#prerequisites)
* [Built With](#built-with)
* [Roadmap](#roadmap)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## About

A tool for the characterization of cloud particle images from the (<a href="http://www.specinc.com/cloud-particle-imager">CPI</a>) probe.  Probe images are classified into 9 categories:

Category            Description
--------            -----------
  Aggregate           A combination of several different monomers with a spread out shape in comparison to a compact irregular. Largely irregular.
  Blank               No target or contrast spanning typically only a few pixels.
  Blurry/Fragment     Indistinguishable frame due to motion or a particle that largely intersects the image border, is too small to distinguish, or is too pixelated.
  Budding Rosette     A rosette shape with short branches.
  Bullet Rosette      A rosette shape with long branches.
  Column              Rectangular with one axis longer than the other and no signs of aggregation.
  Compact Irregular   A small, condensed particle with few to no distinguishable monomers that has an arbitrary shape.
  Plate               Hexagonal and largely independent of aggregation unless the secondary particle is relatively insignificant in size. Exhibits a smooth surface but can have transparent regions.
  Rimed               Evidence of the collection of supercooled water droplets visually apparent as jagged edges. Can obtain any shape.
  Sphere              Circular water droplets


<img src="https://github.com/vprzybylo/cocpit/blob/master/logo.png" alt="Logo" width="150" height="150">

### Additional descriptors that can be processed on the images:

* found in cocpit/pic.py

  Predictor              Description
  ---------              ------------
  Height                 Height of the image in pixels.
  Width                  Width of the image in pixels.
  Laplacian              The variance (standard deviation squared) of a convolved matrix across a single channel image. Lower values are "blurry".
  Number of Contours     Number of contours or closed boundaries from a black and white thresholded image.
  Edges                  Number of edges or localized pixels that meet a contrast gradient requirement in a binary image using the Canny edge detection algorithm.
  $\sigma$ of Edges      Standard deviation or spread in image edge locations.
  Contour Area           Area of the largest contour of a binary image in pixels.
  Contrast               Standard deviation of the image pixel values.
  Circularity            $4*\pi*area/(perimeter^2)$.
  Solidity               Area of a convex hull surrounding the largest contour divided by the largest contour area.
  Complexity             How intricate the particle is. See [@Schmitt2010] for the mathematical representation.
  Equivalent Diameter    Diameter of a circle with the same area as the largest contour.
  Convex Perimeter       Perimeter of the convex hull of the largest contour.
  Hull Area              Area of a convex hull surrounding the largest contour.
  Perimeter              Arc-length of the largest contour.
  Aspect Ratio           Aspect ratio calculated from a rectangle always $\le$ 1.
  Cutoff                 Percentage of pixels that intersect the border or perimeter of the image.
  Extreme Points         How separated the outer most points are on the largest contour using the minimum and maximum x and y extent. The standard deviation in these points represents the spread of the particle.
  Area Ratio             The area of the largest contour divided by the area of an encompassing circle - useful for spheres that have reflection spots that are not captured by the largest contour and leave a horseshoe pattern
  Roundness              Similar to circularity but divided by the convex perimeter that surrounds the largest contour squared instead of the actual convoluted perimeter.
  Perimeter-Area Ratio   Perimeter divided by the area of the largest contour.

## Supported campaigns

**Campaign**       **Location**                       **Date**            **Aircraft**             **\# of images**
  ------------------ ---------------------------------- ------------------- ------------------------ ------------------
  **AIRS-II**        Ottawa, Canada                     Nov 2003-Feb 2004   NSF C-130                504,675
  **ARM IOP**        Western U.S.                       Mar 2000            UND Citation             448,266
  **ATTREX**         Western Pacific                    Feb--Mar 2014       Global Hawk              129,232
  **CRYSTAL-FACE**   Southern Florida                   Jul 2002            NASA's WB-57             225,439
  **CRYSTAL-FACE**   Southern Florida                   Jul 2002            UND Citation             2,483,54
  **Ice-L**          Colorado and Wyoming               Nov-Dec 2007        NSF C-130                1,794,81
  **IPHEX**          Southern Appalachian Mountains     May-June 2014       UND Citation             1,593,537
  **MACPEX**         Central North America (SGP site)   Mar-Apr 2011        WB-57 & Learjet          80,677
  **MC3E**           Central Oklahoma                   April-June 2011     UND Citation             1,614,03
  **MidCix**         Western U.S.                       Apr-May 2004        NASA's WB-57 & Learjet   515,895
  **MPACE**          Alaska                             Sept-Oct 2004       UND Citation             1,177,74
  **OLYMPEX**        Olympic Mountains                  Nov 2015-May 2016   UND Citation             1,308,02
  **POSIDON**        Western Pacific                    Oct 2016            NASA's WB-57             210,497

**TOTAL IMAGES: 12,086,384**


# Installation

1. Clone the repo <br>
	git clone: [git@github.com:vprzybylo/cocpit.git](git@github.com:vprzybylo/cocpit.git)<br/>
3. docker image on docker hub under: vprzybylo/cocpit:v1.2.0 <br>
note to run ./__main__.py requires raw image files (multiple GB per campaign) - contact for image request


## Prerequisites
see requirements.txt

### Built With

* <a href="https://www.python.org/"><a href="https://pytorch.org/docs/stable/torchvision/models.html">Python</a> </a> 
* <a href="https://pytorch.org/docs/stable/torchvision/models.html">Pytorch</a> 
* <a href="http://www.specinc.com/sites/default/files/software_and_manuals/CPI_Post Processing Software Manual_rev1.2_20120116.pdf">cpiview</a> 
  * Desktop-developed software will need to be used to extract ''sheets'' of CPI images from region of interest (ROI) files output from the CPI probe should new data be wished to be processed and classified.
* <a href="https://www.nvidia.com/en-us/">nvidia</a> 
  * Resources used: NVIDIA DGX-1 server utilizing Tesla V100 GPUs. This system is housed in the University at Albanys Tier-3 Data Center, and managed/maintained by the xCITE (ExTREME Collaboration, Innovation and TEchnology) laboratory. The base DGX-1 V100 system contains 8 Tesla V100 GPUs with a combined total of 40,960 CUDA (graphics) cores, 5120 Tensor cores, and 256GB of GPU memory, all linked by NVIDIAs 300GB/s NVLINK interconnect. The DGX-1 is optimized for data loading, data transformations, and training, which are all critical to the ML processes required by this project.


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


