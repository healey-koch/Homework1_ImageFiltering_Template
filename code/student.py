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
    #print(img.shape)
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
    #print(len(image.shape))
    if(len(image.shape) > 2):
        width, height, depth = image.shape
    else:
        width, height = image.shape
        depth = 1
    print(image.shape)
    filtered_image = np.zeros(image.shape)
    #print(image.shape)
    img_arr = list(())
    if(depth > 1):
        for i in range(depth):
            curr_channel = image[:,:,i]
            img_arr.append(curr_channel)
    else:
        img_arr.append(image)
    

    if ((len(kernel) % 2 == 0) or (len(kernel[0]) % 2 == 0)) :
        raise Exception("Sorry, kernel cannot have any even dimensions!")
    kernel_width = len(kernel)
    kernel_height = len(kernel[0])  
    zero_width = (kernel_width - 1) // 2
    zero_height = (kernel_height - 1) // 2
    print("zero_width is " + str(zero_width) + " and height is " + str(zero_height))
    ##################
    # Your code here #

    for q in range(depth):
        print(str(q) + " / " + str(depth))
        curr_channel = pad_image(img_arr[q],zero_width,zero_height)
        new_channel = curr_channel.copy()
        #print(new_channel.shape)
        
        for d in range(width * height):
            #print(str(d) + " / " + str(width * height))
            x,y = index_to_pixel(d,width)
            currPixel = 0
            for j in range(kernel_height * kernel_width):
                kX = j % kernel_width
                kY = (j - kX)//kernel_width
                currPixel += curr_channel[x + kX - zero_width][y + kY - zero_height] * kernel[kX][kY]
            new_channel[x][y] = currPixel
        #print("!!!!!!")
        #print(new_channel.shape)
        if(zero_width > 0 or zero_height > 0):
            print("start transfer is at " + str(zero_width) + " and end transfer is " + str(width-zero_width))
            img_arr[q] = new_channel[zero_width:width + zero_width, zero_height:height + zero_height]
        else:

            img_arr[q] = new_channel

    print(img_arr[0].shape)
    if(depth > 1):
        filtered_image = np.stack(img_arr, axis=2)
    else:
        filtered_image = img_arr[0]
    print(filtered_image.shape)
    ##################
    #print(filtered_image.shape)
    #print(str(width) + " " + str(height))
    print("4")
    return filtered_image
def doShit():
    test_image = io.imread("./data/dog.bmp")
    identity_filter = np.asarray(
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    identity_image = my_imfilter(test_image, identity_filter)
    plt.imshow(identity_image, cmap='gray')
    plt.show()

#doShit()

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
    high_frequencies = np.subtract(image2, my_imfilter(image2, kernel)) # Replace with your implementation
    print("high")

    # (3) Combine the high frequencies and low frequencies, and make sure the hybrid image values are within the range 0.0 to 1.0
    # Your code here
    hybrid_image = (low_frequencies + high_frequencies)# # Replace with your implementation
    hybrid_image = np.clip(hybrid_image,0,255)
    print("hybrid")

    return low_frequencies, high_frequencies, hybrid_image
"""
image1 = io.imread("./data/gokuSmall.bmp")
print(image1.shape)
image2 = io.imread("./data/hatsunemikuSmall.bmp")
print(image2.shape)
cutoff_frequency = 1
low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(
        image1, image2, cutoff_frequency)
print(hybrid_image)
plt.imshow(hybrid_image)
plt.show()
"""