# GeoFilters

geofilter.Filter class contains filter methods for 2D matrices 
to analyse geomorphological structures in digital terrain models. 

Many of the filters have been implemented following the description by 

Florinsky, I. V. \(2017\). An illustrated introduction to general geomorphometry. Progress in Physical Geography, 41\(6\), 723-752.

Kokalj, Z. et al. \(2016\). Relief
visualization toolbox, ver. 2.2. 1 manual. Remote Sensing , 3 \(2\), 398-415.

The Simple Local Relief Models have been implemented using OpenCV convolution functions in order to speed up the computation process. 


## Required packages

Tested on the versions:

numpy == 1.22.3

scikit-learn == 1.1.3

matplotlib == 3.5.3

cv2 == 4.6.0

For the use of the utils.geotif_to_matrix function \(not recommended\): 

gdal == 3.4.1

## Usage 

Just copy the geofilter.py file into your project. 

### Parameters for the Constructor
methods: list, contains the filter methods you want to apply.  

kernel_size: tuple, only relevant for convolution based filters. 

ksize_median: integer, only for slrm_cv2_median. 

azimuth: integer, only for reflectance and insolation. 

elevation: integer, only for reflectance and insolation.     

clip_range: tuple, only for cv2_lmhd and cv2_lmhd_blurred. 

pca_components: integer, only for pca and pca_relief. 

ffr_boundaries: tuple: only for fourier_frequency_removal.

### Parameters for the fit method
matrix: a numpy 2D array.

### Returns of the fit method 

A dictionary with the outputs of the transformation. 

Keys are the method names.

### Example 
```
from geofilter import Filter 

matrix = numpy.random.rand(100,100) 

f = Filter(methods=["slope"])
f.fit(matrix) 

filtered_matrix = f.transformed["slope"]
```

### Available filters

accumulation_curvature

aspect

curvedness

cv2_lmhd \(experimental\)

cv2_lmhd_blurred \(experimental\)

cv2_sobel2D

difference_curvature

eastwardness

fourier_frequency_removal \(experimental\)

gaussian_curvature

horizontal_curvature

horizontal_curvature_deflection

horizontal_excess_curvature

insolation

laplacian

lmhd \(experimental\)

maximal_curvature

mean_curvature

minimal_curvature

nabla

normalized_arctan_nabla \(experimental\)

northwardness

pca_relief \(experimental\)

plan_curvature

reflectance

ring_curvature

rotor

shape_index

slope

slrm_cv2_average

slrm_cv2_gaussian

slrm_cv2_median

unsphericity_curvature

usc

vertical_curvature

vertical_curvature_deflection

vertical_excess_curvature