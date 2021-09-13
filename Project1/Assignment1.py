# Assignment 1 For Image Processing

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
from skimage.util import img_as_ubyte
from skimage import measure
from skimage import color, morphology


def make_im_grey(img):
    """
    Takes an image and converts it to grey scale with values from 0 to 1

    Parameters
    ----------
    img : np.array
        A 3D array of an image with three layers representing RGB values.

    Returns
    -------
    scaled_image : np.array
        A 2D array of values represnting the image converted to grey scale with values
        from 0 to 1.

    """
    greyer = np.array([0.6, 0.3, 0.1])
    unscaled_image = np.dot(img, greyer)
    scaled_image = unscaled_image / np.max(unscaled_image)
    return scaled_image


def display_img(img):
    """
    Displays the image in gray scale.     

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    io.imshow(img, cmap='gray')
    io.show()
def make_histogram(img, bins):
   
    max_value = np.max(img)
    min_value = np.min(img)
    
    step_size = 1 / bins
    
    curr_value = min_value
    
    total_pixels = img.shape[0] * img.shape[1]
    
    hist = np.zeros((bins, 2))
    for i in range(0, bins):
        in_curr_bin =((img >= curr_value) & (img < curr_value +step_size)).sum() / total_pixels
    
        hist[i][0] = curr_value
        hist[i][1] = in_curr_bin
    
        curr_value = curr_value + step_size
    
    
    return hist

def display_histogram(hist, bins):
    plt.bar(hist[:,0], hist[:,1], align='edge', width=1/bins)
    
    
def otsu_threshold_finder(hist):
    
    T = 0
    curr_best = -float('inf')
    for t in range(1, hist.shape[0]):
        # if you start at 0 or go to the end you get 0's in the denominator
        w0 = hist[:t,1].sum()
        mu0 = (hist[:t,1] * hist[:t,0]).sum()/w0

    
        w1 = hist[t:,1].sum()
        mu1 = (hist[t:,1] * hist[t:,0]).sum()/w1
        
        var_b = w0*w1 * (mu0 - mu1)**2
        
        if var_b > curr_best:
            T = t
            curr_best = var_b
            
    return hist[T][0]
        
        
def threshold_image(img, threshold):
    thresh = np.copy(img)
    thresh[img>threshold] = 1
    thresh[img<threshold] = 0
    return thresh
     

puppy_file = f"D:\School\Fall 2021\ImageProcessing\ImageProcessing\Images\puppy.jpg"

puppy_image = io.imread(puppy_file)

grey_puppy = make_im_grey(puppy_image)

display_img(grey_puppy)

# %%
bins = 100 



hist = make_histogram(grey_puppy, bins)
display_histogram(hist, bins)

# %% 
puppy_threshold = otsu_threshold_finder(hist)

thresholded_puppy = threshold_image(grey_puppy, puppy_threshold)

display_img(thresholded_puppy)

#display_img(grey_puppy)
# %%

total_pixels = grey_puppy.shape[0] * grey_puppy.shape[1]

cdf = np.cumsum(hist[:,1])


cdf_indices = np.round(grey_puppy * 99, 0).astype(int)


adapted_img = cdf[cdf_indices]

display_img(adapted_img)

# %% 
level_hist = make_histogram(adapted_img, bins)
display_histogram(level_hist, bins)
# %% 
hist = make_histogram(thresholded_puppy, bins)
display_histogram(hist, bins) 

# %% 


removed = morphology.remove_small_objects(np.round(thresholded_puppy).astype(int).astype(bool), 1000)

new_regions = measure.label(removed.astype(int))
io.imshow(new_regions)

# %% 


def save_histogram(hist, bins, title, fname):
    plt.figure()
    plt.bar(hist[:,0], hist[:,1], align='edge', width=1/bins)
    plt.title(title)
    plt.savefig(fname)
    

def histogram_adaptation(img, hist, bins):
        
    cdf = np.cumsum(hist[:,1])
    cdf = cdf/cdf[-1] # there were issues with machine precision.
    
    cdf_indices = np.round((img) * (bins-1), 0).astype(int)
    
    
    adapted_img = cdf[cdf_indices]
    
    return adapted_img
    
    
puppy_file = f"D:\School\Fall 2021\ImageProcessing\ImageProcessing\Images\Puppy.jpg"
fire_file = f"D:\School\Fall 2021\ImageProcessing\ImageProcessing\Images\Fire.jpg"
florence_file = f"D:\School\Fall 2021\ImageProcessing\ImageProcessing\Images\Florence.jpg"
rome_file = f"D:\School\Fall 2021\ImageProcessing\ImageProcessing\Images\Rome.jpeg"

folder_path = f"D:\School\Fall 2021\ImageProcessing\ImageProcessing\Images"


FileNameBases = ["\\Puppy", "\\Fire", 
                 "\\Florence", "\\Rome"]

files = [puppy_file, fire_file, florence_file, rome_file]

file_type = '.png'
bins = 100
for i in range(0, len(files)):

    file = files[i]
    img = io.imread(file)
    
    grey_img = make_im_grey(img)
    
    
    io.imsave(folder_path+FileNameBases[i]+'_Grey'+file_type, img_as_ubyte(grey_img))
    
    hist = make_histogram(grey_img, bins)
    
    title = FileNameBases[i].replace('\\','') + ' Grey Histogram'
    grey_histogram_file = folder_path+FileNameBases[i]+'_Grey_Histogram'+file_type
    
    save_histogram(hist, bins, title, grey_histogram_file)

    threshold = otsu_threshold_finder(hist)

    thresholded_img = threshold_image(grey_img, threshold)

    io.imsave(folder_path+FileNameBases[i]+'_Thresholded'+file_type, img_as_ubyte(thresholded_img))
    
    threshold_hist = make_histogram(thresholded_puppy, bins)
    
    title = FileNameBases[i].replace('\\','') + ' Thresholded Histogram'
    thresholded_histogram_file = folder_path+FileNameBases[i]+'_Thresholded_Histogram'+file_type
    
    save_histogram(threshold_hist, bins, title, thresholded_histogram_file)
    
    adapted_img = histogram_adaptation(grey_img, hist, bins)
    io.imsave(folder_path+FileNameBases[i]+'_Histogram_Adapted'+file_type, img_as_ubyte(adapted_img))
    
    
    title = FileNameBases[i].replace('\\','') + 'Histogram Adapted Histogram'
    adapted_histogram_file = folder_path+FileNameBases[i]+'Histogram_Adapted_Histogram'+file_type
    
    level_hist = make_histogram(adapted_img, bins)
    save_histogram(level_hist, bins, title, adapted_histogram_file)
    
    
    removed = morphology.remove_small_objects(np.round(thresholded_img).astype(int).astype(bool), 100)

    new_regions = measure.label(removed.astype(int))
   
    plt.imsave(folder_path+FileNameBases[i]+'_Regions'+file_type, new_regions)

# %% 

from skimage import exposure
fire_file = f"D:\School\Fall 2021\ImageProcessing\ImageProcessing\Images\Fire.jpg"
fire_img = io.imread(fire_file)
   
grey_fire = make_im_grey(fire_img)

for i in [2, 4, 8, 16, 32, 64, 128, 256]:
        
    
    equalized_pup = exposure.equalize_hist(grey_puppy, i)
    to_save = folder_path + '\\Puppy_Histogram_Equalize_' + str(i) + '.png'
    
    io.imsave(to_save, img_as_ubyte(equalized_pup))
    
    
    title = 'Puppy Histogram Adaptation with ' + str(i) + ' bins'
    adapted_histogram_file = folder_path + '\\Puppy_Histogram_Equalize_Histogram_' + str(i) + '.png'
    
    adapted_hist = make_histogram(equalized_pup, bins)
    
    save_histogram(adapted_hist, bins, title, adapted_histogram_file)
    
    
    
    equalized_fire = exposure.equalize_hist(grey_fire, i)
    to_save = folder_path + '\\Fire_Histogram_Equalize_' + str(i) + '.png'
   
    
    title = 'Fire Histogram Adaptation with ' + str(i) + ' bins'
    adapted_histogram_file = folder_path + '\\Fire_Histogram_Equalize_Histogram_' + str(i) + '.png'
    
    adapted_hist = make_histogram(equalized_fire, bins)
    
    save_histogram(adapted_hist, bins, title, adapted_histogram_file)
    
    
    io.imsave(to_save, img_as_ubyte(equalized_fire))
    
# %% 
from skimage.morphology import disk
from skimage.filters import rank

for i in [32, 64, 128, 256, 512, 1024, 2048]:       
    footprint = disk(i)
    
    img = img_as_ubyte(grey_fire)
    img_eq = rank.equalize(img, footprint)
    
    to_save = folder_path + '\\Fire_Local_Histogram_Equalize_' + str(i) + '.png'
    
    io.imsave(to_save, img_eq)
    
# %% 
footprint = disk(128)

img = img_as_ubyte(grey_puppy)
img_eq = rank.equalize(img, footprint)
   
io.imshow(img_eq)

img = img_as_ubyte(grey_puppy)
img_eq = rank.equalize(img, footprint)

to_save = folder_path + '\\Puppy_Local_Histogram_Equalize_' + str(i) + '.png'

io.imsave(to_save, img_eq)

# %% 
xray_file = f"D:\School\Fall 2021\ImageProcessing\ImageProcessing\Images\\xray.png"
xray_img = io.imread(xray_file)

grey_xray = make_im_grey(xray_img)

for i in [32, 64, 128, 256, 512, 1024, 2048]:       
    footprint = disk(i)
    
    img = img_as_ubyte(grey_xray)
    img_eq = rank.equalize(img, footprint)
    
    to_save = folder_path + '\\Xray_Local_Histogram_Equalize_' + str(i) + '.png'
    
    io.imsave(to_save, img_eq)
    
# %%

io.imsave(folder_path+'\Xray_Grey.png', img_as_ubyte(grey_xray))

