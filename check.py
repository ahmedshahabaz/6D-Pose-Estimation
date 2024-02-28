import numpy as np

import yaml
from scipy.spatial.transform import Rotation as R
#import pyrealsense2 as rs
import os
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from PIL import ImageDraw
import glob
import matplotlib.pyplot as pt


ADD_List = []
ADD_model = []
files = glob.glob('*_4.np[yz]')

for i,file in enumerate(files):
	ADD_List = np.append(ADD_List,np.load(file))

#print(ADD_List)


x = np.linspace(0,7,len(ADD_List))
y = np.zeros(x.shape)

for j in range(len(ADD_List)):
	y[j] = len([i for i in ADD_List if i < x[j]])/len(ADD_List)
	
pt.figure(1)
pt.plot(x,y)
pt.plot(np.full(y.shape,3),y)
pt.xlabel('ADD Threshold')
pt.ylabel('ADD Percentage')
pt.title('Average Threshold for Sugar Box ')

pt.show()

np.save("Soup.npy",ADD_List)
