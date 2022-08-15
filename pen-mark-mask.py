#!/usr/bin/env python
# coding: utf-8

# Reading

# In[10]:


import openslide
from openslide import open_slide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff
import cv2
import math
import skimage.morphology
import concurrent.futures


# In[33]:


WSI = open_slide("sample_penmark.tiff")
thumbnail = WSI.get_thumbnail(size = (500,500))
thumbnail.show()
thumb = thumbnail.save("thumb.png")


# In[34]:


img = cv2.imread('thumb.png')
img = cv2.GaussianBlur(img, (3,3), 0)
plt.imshow(img)


# In[35]:


img_tohsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(img_tohsv)


# In[36]:


tissue = cv2.inRange(img_tohsv, np.array([135, 10, 30]), np.array([170, 255, 255]))
tissue_mask = tissue
plt.imshow(tissue)


# In[37]:


black_marker = cv2.inRange(img_tohsv, np.array([0, 0, 0]), np.array([180, 255, 125]))
red_marker = cv2.inRange(img_tohsv, np.array([0,50,50]), np.array([10,255,255]))
blue_marker = cv2.inRange(img_tohsv, np.array([100, 125, 30]), np.array([130, 255, 255]))
green_marker = cv2.inRange(img_tohsv, np.array([40, 125, 30]), np.array([70, 255, 255]))
hsv_mask = cv2.bitwise_or(cv2.bitwise_or(cv2.bitwise_or(red_marker, blue_marker), green_marker),black_marker)
plt.imshow(hsv_mask)


# In[38]:


dilate_radius = 3
_mask_dilated = cv2.dilate(hsv_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_radius,dilate_radius)))


# In[39]:


plt.imshow(_mask_dilated)


# In[ ]:





# In[ ]:




