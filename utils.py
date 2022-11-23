import numpy as np

"""
These functions have not been properly tested and there might be better options.


Copyright: Michael Linke 

License: 
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) 
https://creativecommons.org/licenses/by-nc/4.0/

"""


def get_xyz_data_from_file(filename):
    """
    Loads the content of a XYZ file into a list. 
    Every list element has 3 elements: [x, y, z].
    """

    d = open(filename)
    inhalt = d.read()
    d.close()
    zeilen = inhalt.split("\n")
    xyz_data = []
    for i in zeilen:    
        string_data = i.split(" ")
        
        if len(string_data) == 3:            
            xyz_data.append([float(string_data[0]), float(string_data[1]), float(string_data[2])])
    return xyz_data


def xyz_to_matrix(xyz_data, resolution):
    """Converts the output of get_xyz_data_from_file into a numpy array."""

    x_min = xyz_data[0][0]
    x_max = xyz_data[0][0]
    y_min = xyz_data[0][1]
    y_max = xyz_data[0][1]

    for x in range(len(xyz_data)):
        if x_min > xyz_data[x][0]:
            x_min = xyz_data[x][0]
        if x_max < xyz_data[x][0]:
            x_max = xyz_data[x][0]
        if y_min > xyz_data[x][1]:
            y_min = xyz_data[x][1]
        if y_max < xyz_data[x][1]:
            y_max = xyz_data[x][1]
    
    matrix_width = int((x_max - x_min)/resolution + 1.5) 
    matrix_heigth = int((y_max - y_min)/resolution + 1.5) 
    
    new_matrix = np.zeros([matrix_width, matrix_heigth])

    for x in range(len(xyz_data)):
        i = int((1/resolution) * (xyz_data[x][0] - x_min))
        j = int((1/resolution) * (xyz_data[x][1] - y_min))        
        new_matrix[i][j] = xyz_data[x][2]
        
    return new_matrix       


def geotif_to_matrix(filepath, lower=-10000): 
    """Loads the content of a GeoTiF file and converts it into a numpy array."""

    import gdal 
    data = gdal.Open(filepath)
    matrix = np.array(data.GetRasterBand(1).ReadAsArray())
    min_value = 100000000.
    for i in range(matrix.shape[0]): 
        for j in range(matrix.shape[1]): 
            v = matrix[i][j]
            if v < min_value and v > lower: 
                min_value = v 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i][j]
            if v < lower:
                matrix[i][j] = min_value
    matrix -= min_value    
    return matrix