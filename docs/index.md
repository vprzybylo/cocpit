
## ABOUT

A tool for the characterization of cloud particle images from the (CPI) probe. Three models are featured: one to separate cloud drops (liquid) from all other images, one to separate quality images of ice crystals from blurry, fragmented, and blank images, and the last to classify frozen hydrometeors from the remaining good images. Probe images are classified into 11 categories (aggregates, blurry, budding rosettes, bullet rosettes, columns, compact irregulars, fragments, needles, plates, rimed aggregates and rimed columns) using a convolutional neural network.

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

