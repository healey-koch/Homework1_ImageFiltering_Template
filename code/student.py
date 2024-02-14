# Homework 1 Image Filtering Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
import matplotlib.pyplot as plt

def pad_image(image, width_padding, height_padding):
    img = np.pad(image,((width_padding,width_padding),(height_padding,height_padding)), 'constant')
    return img

def index_to_pixel(index, width):
    x = index % width
    y = (index-x)//width
    return (x, y)


def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the homework webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """

    if ((len(kernel) % 2 == 0) or (len(kernel[0]) % 2 == 0)) :
        raise Exception("Sorry, kernel cannot have any even dimensions!")
    img_arr = list(())

    if(len(image.shape) > 2):       #if the image is RGB
        width, height, depth = image.shape
        for i in range(depth):
            curr_channel = image[:,:,i]
            img_arr.append(curr_channel)
    else:                           #if the image is BW
        width, height = image.shape
        depth = 1
        img_arr.append(image)

    zero_width = len(kernel) // 2 #half of the kernel's width
    zero_height = len(kernel[0]) // 2   #half of the kernel's height
    
    for q in range(depth):
        curr_channel = pad_image(img_arr[q],zero_width,zero_height)
        new_channel = curr_channel.copy()
        for y in range(zero_height,height + zero_height):
            for x in range(zero_width, width + zero_width):
                new_channel[x][y] = np.sum(np.multiply(curr_channel[x-zero_width : x + 1 + zero_width, y - zero_height: y + 1 + zero_height], kernel))
        img_arr[q] = new_channel[zero_width:width + zero_width, zero_height:height + zero_height]


    if (depth > 1):
        return np.clip(np.stack(img_arr, axis=2),0,255)
    else:
        return np.clip(img_arr[0],0,255)

def doShit():
    test_image = io.imread("./data/marilyn_gray.bmp")
    identity_filter = np.asarray(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    #identity_filter = np.full((5,5),1/25)
    image = my_imfilter(test_image, identity_filter)
    sobel_image = np.clip(image+127, 0.0, 255)
    plt.imshow(sobel_image, cmap='gray')
    plt.show()

"""
EXTRA CREDIT placeholder function
"""

def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the homework webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here
    low_frequencies = my_imfilter(image1, kernel)# Replace with your implementation
    print("low")

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #

    

    high_frequencies = np.clip(np.subtract(image2, my_imfilter(image2, kernel)), 0, 255) # Replace with your implementation
    print("high")

    

    # (3) Combine the high frequencies and low frequencies, and make sure the hybrid image values are within the range 0.0 to 1.0
    # Your code here
    hybrid_image = np.add(low_frequencies,high_frequencies)# # Replace with your implementation
    #hybrid_image = np.clip(hybrid_image,0,255)
    print("hybrid")

    return low_frequencies, high_frequencies, hybrid_image
def hybrid():
    image1 = io.imread("./data/gokuSmall.bmp")
    image2 = io.imread("./data/hatsunemikuSmall.bmp")
    cutoff_frequency = 7
    low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(
            image1, image2, cutoff_frequency)
    #print("low freq is " + str(low_frequencies[140][25]))
    #print("high freq is " + str(high_frequencies[140][25]))
    #print("hybrid is " + str(hybrid_image[140][25]))
    plt.imshow(low_frequencies)
    plt.show()
    plt.imshow(high_frequencies)
    plt.show()
    plt.imshow(hybrid_image)
    plt.show()

#hybrid()
doShit()