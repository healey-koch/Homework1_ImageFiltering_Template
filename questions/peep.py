import numpy as np
img = np.full((320,640,3),255)
img = np.pad(img,((2,2),(3,3),(0,0)), 'constant')
print(img.shape)