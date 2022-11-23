"""

Filter class contains filter methods for 2D matrices 
to analyse geomorphological structures in digital terrain models. 

Many of the filters have been implemented following the description by 


Florinsky, I. V. (2017). An illustrated introduction to general geomorphometry. Progress in Physical Geography, 41(6), 723-752.

Kokalj, Z. et al. (2016). Relief visualization toolbox, ver. 2.2. 1 manual. Remote Sensing , 3 (2), 398-415.

Some others are experimental, e.g. the lmhd filters (local mean height difference), 
a slrm-like filter for preprocessing inputs of artificial neural networks. 

Tested only on small matrices. The transformation of a 1500x1500 matrix cost ca. 1 GB memory. 

Copyright: Michael Linke 

License: 
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) 
https://creativecommons.org/licenses/by-nc/4.0/

"""

import numpy as np



class Filter: 

    def __init__(self, 
            methods=[],            
            kernel_size=(41, 41), 
            ksize_median=41,           
            azimuth=315, 
            elevation=35,             
            clip_range=(-0.1, 0.1), 
            pca_components=0, 
            ffr_boundaries=(-10000, 10000)):    

        self.methods = methods
        self.kernel_size = kernel_size,
        self.ksize_median = ksize_median
        self.azimuth = np.deg2rad(azimuth) 
        self.elevation = np.deg2rad(elevation)              
        self.clip_range = clip_range 
        self.pca_components = pca_components
        self.ffr_boundaries = ffr_boundaries

        self.transform_dict = {
            "slrm_cv2_average": self.slrm_cv2_average, 
            "slrm_cv2_gaussian": self.slrm_cv2_gaussian,
            "slrm_cv2_median": self.slrm_cv2_median,                 
            "cv2_sobel2D": self.cv2_sobel2D, 
            "northwardness": self.northwardness, 
            "eastwardness": self.eastwardness,                
            "lmhd": self.local_mean_height_difference, 
            "cv2_lmhd": self.cv2_lmhd, 
            "cv2_lmhd_blurred": self.cv2_lmhd_blurred, 
            "pca_relief": self.pca_relief,
            "slope": self.slope, 
            "aspect": self.aspect,             
            "plan_curvature": self.plan_curvature, 
            "horizontal_curvature": self.horizontal_curvature, 
            "vertical_curvature": self.vertical_curvature, 
            "difference_curvature": self.difference_curvature, 
            "accumulation_curvature": self.accumulation_curvature, 
            "ring_curvature": self.ring_curvature, 
            "rotor": self.rotor, 
            "horizontal_curvature_deflection": self.horizontal_curvature_deflection, 
            "vertical_curvature_deflection": self.vertical_curvature_deflection,
            "mean_curvature": self.mean_curvature, 
            "gaussian_curvature": self.gaussian_curvature, 
            "minimal_curvature": self.minimal_curvature, 
            "maximal_curvature": self.maximal_curvature, 
            "unsphericity_curvature": self.unsphericity_curvature,
            "usc" : self.unsphericity_curvature, 
            "horizontal_excess_curvature": self.horizontal_excess_curvature, 
            "vertical_excess_curvature": self.vertical_excess_curvature, 
            "laplacian": self.laplacian, 
            "shape_index": self.shape_index, 
            "curvedness": self.curvedness, 
            "reflectance": self.reflectance, 
            "insolation": self.insolation, 
            "nabla": self.nabla,
            "normalized_arctan_nabla": self.normalized_arctan_nabla, 
            "fourier_frequency_removal": self.fourier_frequency_removal
        }


    def slrm_cv2_average(self):         
        from cv2 import blur 

        blurred = blur(self.matrix, self.kernel_size[0]) 
        return self.matrix - blurred 


    def slrm_cv2_gaussian(self):         
        from cv2 import GaussianBlur 

        blurred = GaussianBlur(self.matrix, self.kernel_size[0], sigmaX=0) 
        return self.matrix - blurred 


    def slrm_cv2_median(self):         
        from cv2 import medianBlur 

        matrix = self.matrix.copy()         
        matrix -= np.min(matrix) 
        matrix /= np.max(matrix) 
        matrix *= 255 
        
        m = matrix.copy()        
        m = m.astype('uint8')
        matrix = matrix.astype('uint8')
        blurred = medianBlur(m, self.ksize_median) 
        blurred = blurred.astype(self.matrix.dtype)         
        return matrix - blurred        


    def cv2_sobel2D(self): 
        from cv2 import Sobel, CV_64F 
        x = Sobel(self.matrix, CV_64F, 1, 0, ksize=5)
        y = Sobel(self.matrix, CV_64F, 0, 1, ksize=5)
        return np.sqrt(x ** 2 + y ** 2)



    def local_mean_height_difference(self):     
        from cv2 import blur 

        lmhd = blur(self.matrix, (3,3)) 
        lmhd = self.matrix - lmhd
        lmhd = np.clip(lmhd, -0.1, 0.1)
        lmhd = blur(lmhd, (3,3))
        return np.nan_to_num(lmhd)


    def cv2_lmhd(self): 
        from cv2 import filter2D        
        
        kernel = np.array([[1,1,1], 
                           [1,0,1], 
                           [1,1,1]])
        kernel = kernel / 8
        blurred = filter2D(self.matrix, -1, kernel)
        lmhd = self.matrix - blurred         
        lmhd = np.clip(lmhd, self.clip_range[0], self.clip_range[1])        
        return lmhd  



    def cv2_lmhd_blurred(self):
        from cv2 import filter2D, blur
        
        kernel = np.array([[1,1,1], 
                           [1,0,1], 
                           [1,1,1]])
        kernel = kernel / 8
        blurred = filter2D(self.matrix, -1, kernel)
        lmhd = self.matrix - blurred         
        lmhd = np.clip(lmhd, self.clip_range[0], self.clip_range[1])
        lmhd = blur(lmhd, (3,3))
        return lmhd  




    def pca_relief(self): 
        # Uses Principal Component Analysis to reduce information in the matrix 
        # and subtracts the reduced version from the original. 
        from sklearn.decomposition import PCA 

        pca_instance = PCA(self.pca_components) 
        width = self.matrix.shape[0] 
        print(width)        
        f_matrix = self.matrix.copy()
        decomposed = pca_instance.fit_transform(f_matrix)
        inverted = pca_instance.inverse_transform(decomposed) 
        return self.matrix - inverted


    def get_gradients(self): 
        z = np.gradient(self.matrix) 
        p = z[0] 
        q = z[1] 
        grad_p = np.gradient(p) 
        r = grad_p[0] 
        s = grad_p[1] 
        grad_q = np.gradient(q) 
        s2 = grad_q[0] 
        t = grad_q[1]     
        return p, q, r, s, t


    def get_first_gradients(self):     
        z = np.gradient(self.matrix) 
        p = z[0] 
        q = z[1]        
        return p, q


    def get_second_gradients(self):     
        z = np.gradient(self.matrix) 
        p = z[0] 
        q = z[1]  
        grad_p = np.gradient(p) 
        r = grad_p[0]     
        grad_q = np.gradient(q) 
        t = grad_q[1]     
        return r, t      
        

    def get_third_degree_gradients(self): 
        grad_r = np.gradient(self.r) 
        g = grad_r[0] 
        k = grad_r[1] 
        grad_t = np.gradient(self.t) 
        m = grad_t[0] 
        h = grad_t[1] 
        return g, h, k, m 
    

    """
    The following functions with numbers in brackets are implementations of the equations 4 - 28 in: 
    Florinsky, I. V. (2017). An illustrated introduction to general geomorphometry. Progress in Physical Geography, 41(6), 723-752.
    The numbers in brackets refer to the equation number in the article.
    """

    
    def slope(self): 
        # (4)
        G = np.sqrt(self.p**2 + self.q**2) 
        G = np.nan_to_num(G)    
        return np.nan_to_num(np.arctan(G))  


    def aspect(self): 
        # (5)
        A = - 90 * (1 - np.sign(self.q)) * (1 - np.abs(np.sign(self.p))) 
        A += 180 * (1 + np.sign(self.p)) 
        A -= (180 / np.pi) * np.sign(self.p) * np.arccos(-self.q / np.sqrt(self.p**2 + self.q**2)) 
        A = np.nan_to_num(A)
        return A 


    def northwardness(self): 
        # (6)
        A = self.aspect()
        return np.cos(A) 


    def eastwardness(self): 
        # (7)
        A = self.aspect()
        return np.sin(A) 


    def plan_curvature(self):
        # (8) 
        numerator = self.q**2 * self.r - 2*self.p*self.q*self.s + self.p**2 * self.t  
        numerator *= (-1) 
        denominator = np.sqrt((self.p**2 + self.q**2)**3) 
        return np.nan_to_num(numerator / denominator)  


    def horizontal_curvature(self): 
        # (9)
        p, q, r, s, t = self.p, self.q, self.r, self.s, self.t
        numerator = q**2 * r - 2*p*q*s + p**2 * t 
        numerator *= (-1) 
        denominator = (p**2 + q**2) * np.sqrt(1 + p**2 + q**2) 
        return np.nan_to_num(numerator / denominator)  


    def vertical_curvature(self): 
        # (10)
        p, q, r, s, t = self.p, self.q, self.r, self.s, self.t
        numerator = p**2 * r + 2*p*q*s + q**2 * t 
        numerator *= (-1) 
        denominator = (p**2 + q**2) * np.sqrt((1 + p**2 + q**2)**3) 
        A = numerator / denominator 
        A = np.nan_to_num(A)
        return A


    def difference_curvature(self): 
        # (11)
        return np.nan_to_num((1/2) * (self.vertical_curvature() - self.horizontal_curvature()) ) 


    def accumulation_curvature(self): 
        # (14)
        return np.nan_to_num(self.horizontal_curvature() * self.vertical_curvature())  


    def ring_curvature(self): 
        # (15)
        p, q, r, s, t = self.p, self.q, self.r, self.s, self.t
        numerator = (p**2 - q**2) * s - p*q*(r - t) 
        denominator = (p**2 + q**2) * (1 + p**2 + q**2) 
        return np.nan_to_num((numerator / denominator)**2)  


    def rotor(self): 
        # (16)
        p, q, r, s, t = self.p, self.q, self.r, self.s, self.t
        numerator = (p**2 - q**2) * s - p*q*(r - t) 
        denominator = np.sqrt((p**2 + q**2)**3) 
        return np.nan_to_num(numerator / denominator)  

    
    def horizontal_curvature_deflection(self): 
        # (17)
        p, q, _, _, _, g, h, k, m = self.p, self.q, self.r, self.s, self.t, self.g, self.h, self.k, self.m
        D = q**3 * g - p**3 * h + 3*p*q*(p*m - q*k) 
        D /= np.sqrt((p**2 + q**2)**3 * (1 + p**2 + q**2)) 
        T = 2 + 3*(p**2 + q**2) 
        T /= (1 + p**2 + q**2)    
        D -= self.horizontal_curvature() * self.rotor() * T 
        return np.nan_to_num(D)  


    def vertical_curvature_deflection(self): 
        # (18)
        p, q, r, _, t, g, h, k, m = self.p, self.q, self.r, self.s, self.t, self.g, self.h, self.k, self.m
        D = q**3 * m - p**3 * k + 2*p*q*(q*k - p*m) - p*q*(q*h - p*g) 
        D /= np.sqrt( (p**2 + q**2)**3 * (1 + p**2 + q**2)**3) 
        a = 2 * (r + t) / np.sqrt((1 + p**2 + q**2)**3) 
        b = (2 + 5*(p**2 + q**2)) / (1 + p**2 + q**2)
        D -= self.rotor() * (a + self.vertical_curvature() * b)
        return np.nan_to_num(D)  


    def mean_curvature(self): 
        # (21)
        p, q, r, s, t = self.p, self.q, self.r, self.s, self.t
        numerator = - ( (1 + q**2)*r - 2*p*q*s + (1 + p**2)*t )    
        denominator = 2 * np.sqrt((1 + p**2 + q**2)**3)     
        return np.nan_to_num(numerator / denominator)            


    def gaussian_curvature(self): 
        # (22)
        p, q, r, s, t = self.p, self.q, self.r, self.s, self.t
        numerator = r*t - s**2 
        denominator = (1 + p**2 + q**2)**2 
        return np.nan_to_num(numerator / denominator) 


    def minimal_curvature(self): 
        # (19)
        H = self.mean_curvature()
        K = self.gaussian_curvature()
        return np.nan_to_num(H - np.sqrt(H**2 - K)) 


    def maximal_curvature(self): 
        # (20)
        H = self.mean_curvature()
        K = self.gaussian_curvature()
        return np.nan_to_num(H + np.sqrt(H**2 - K)) 


    def unsphericity_curvature(self):
        # (23)
        H = self.mean_curvature()
        K = self.gaussian_curvature()
        return np.nan_to_num(np.sqrt(H**2 - K)) 


    def horizontal_excess_curvature(self): 
        # (12)
        M = self.unsphericity_curvature()
        E = self.difference_curvature()
        A = M - E
        A = np.nan_to_num(A)
        return A


    def vertical_excess_curvature(self): 
        # (13)        
        M = self.unsphericity_curvature()
        E = self.difference_curvature()
        return np.nan_to_num(M + E) 


    def laplacian(self): 
        # (24)
        return np.nan_to_num(self.r + self.t)  


    def shape_index(self): 
        # (25)
        k_min = self.minimal_curvature() 
        k_max = self.maximal_curvature() 
        return np.nan_to_num((2/np.pi) * np.arctan((k_max + k_min) / (k_max - k_min)))  


    def curvedness(self): 
        # (26)
        k_min = self.minimal_curvature() 
        k_max = self.maximal_curvature() 
        C = (k_max**2 + k_min**2) / 2 
        return np.nan_to_num(np.sqrt(C))  


    def cot(self, x): 
        # cotangent
        return np.nan_to_num(np.cos(x) / np.sin(x))  


    def reflectance(self): 
        # (27)
        # a = azimuth 
        # e = elevation        
        from numpy import sin, cos

        p, q, a, e = self.p, self.q, self.azimuth, self.elevation
        R = 1 - p*sin(a) * self.cot(e) - q*cos(a) * self.cot(e) 
        R /= (np.sqrt(1 + p**2 + q**2) * np.sqrt(1 + (sin(a) * self.cot(e)**2 + (cos(a) * self.cot(e))))) 
        return np.nan_to_num(R)  


    def insolation(self): 
        # (28)
        # a = azimuth 
        # e = elevation        
        from numpy import sin, cos

        p, q, a, e = self.p, self.q, self.azimuth, self.elevation
        I = 50 * ( 1 + np.sign(  sin(e) - cos(e) * (p*sin(a) + q*cos(a)))) 
        I *= ((sin(e) - cos(e) * (p*sin(a) + q*cos(a)))  /  (np.sqrt(1 + p**2 + q**2))) 
        return np.nan_to_num(I)   


    """The following filters are experimental stuff that should be used with caution."""


    def nabla(self):  
        # nabla operator   
        ps = self.p + self.q 
        return np.nan_to_num(ps) 


    def arctan_nabla(self): 
        nab = self.nabla() 
        return np.nan_to_num(np.arctan(nab))


    def normalized_arctan_nabla(self): 
        nab = self.arctan_nabla() 
        return 2. * nab / np.pi


    def fourier_frequency_removal(self): 
        from numpy.fft import rfft2, irfft2        

        ft = rfft2(self.matrix.copy())
        log_ft = np.log(ft)        
        ft[log_ft > self.ffr_boundaries[1]] = 0 
        ft[log_ft < self.ffr_boundaries[0]] = 0 
        ift = irfft2(ft) 
        return ift


    def invalid_method(self): 
        raise Exception("Not a valid method: " + self.method) 


    def fit(self, matrix:np.ndarray):

        if type(matrix) != np.ndarray or len(matrix.shape) != 2: 
            raise Exception("Input to fit method has to be a 2D numpy array!")
        

        self.transformed = {}
        self.is_transform_dict_advanced = False
        self.matrix = matrix 
        self.p, self.q, self.r, self.s, self.t = self.get_gradients()
        self.g, self.h, self.k, self.m = self.get_third_degree_gradients() #self.r, self.t

        if type(self.methods) == str: 
            self.methods = [self.methods]
        elif type(self.methods) != list: 
            raise Exception("methods argument must be a string or a list of strings!")  
        
        for method in self.methods: 
            self.method = method
            t = self.transform_dict.get(self.method, self.invalid_method)
            t = self.transform_dict[self.method]()

            self.transformed[self.method] = t  