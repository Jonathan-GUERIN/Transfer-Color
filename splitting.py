# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 21:40:42 2022

@author: jonathan
"""

#import os

#os.system('cmd /k "ffmpeg -i C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/input2/fish_soup.mp4 -vf fps=25 C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/input2/frame%06d.jpeg"')

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from PIL import Image
import matplotlib.image as ima
import os
import ffmpeg