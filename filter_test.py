# Test for every filter method provided by the geofilters.Filter class. 


"""
Copyright: Michael Linke 

License: 
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) 
https://creativecommons.org/licenses/by-nc/4.0/
"""


import numpy as np 
import matplotlib.pyplot as plt 
from geofilter import Filter 
import math as m


def set_mound(matrix, radius, height, x_middle, y_middle):
    x_min = x_middle - radius
    x_max = x_middle + radius
    y_min = y_middle - radius
    y_max = y_middle + radius
    box = [x_min, x_max, y_min, y_max]
    for x in range(int(box[0]), int(box[1])):
        for y in range(int(box[2]), int(box[3])):
            a = abs( float(x) - float(x_middle) )
            b = abs( float(y) - float(y_middle) )
            c = m.sqrt(a**2 + b**2)
            if c <= radius:
                h = m.sqrt(radius**2 - c**2)
                h = h * (height/radius)
                matrix[x][y] += h
    return matrix


matrix = np.ones((100, 100)) 
matrix = set_mound(matrix, 25, 10, 50, 50)
matrix = (np.random.rand(100,100) / 10) + matrix 

#matrix = np.random.rand(100,100)

f = Filter(methods=[], pca_components=2, ffr_boundaries=(-3, 10)) 
methods = [key for key in f.transform_dict.keys()]
#methods = [str(methods[-1])]
#methods = ["cv2_lmhd_blurred"]
#methods = ["pca_relief"]
print("Implemented filters: ")
print()
for m in methods: 
    print(m)
print()
f.methods = methods
f.fit(matrix) 

for method in methods: 
    print(method)
    plt.imshow(f.transformed[method], cmap="gray") 
    plt.title(method)
    plt.colorbar()
    plt.show()