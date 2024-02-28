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
from scipy.integrate import simps
from numpy import trapz

mustard = np.load('npy files/000049_3.npy')
soup = np.load('npy files/000049_3_ADDS.npy')
sugar = np.load('Sugar.npy')

x_m = np.linspace(0,7,len(mustard))
x_sp = np.linspace(0,7,len(soup))
x_sg = np.linspace(0,7,len(sugar))

y_m = np.zeros(x_m.shape)
y_sp = np.zeros(x_sp.shape)
y_sg = np.zeros(x_sg.shape)

for j in range(len(mustard)):
	y_m[j] = len([i for i in mustard if i < x_m[j]])/len(mustard)

for j in range(len(soup)):
	y_sp[j] = len([i for i in soup if i < x_sp[j]])/len(soup)


for j in range(len(sugar)):
	y_sg[j] = len([i for i in sugar if i < x_sg[j]])/len(sugar)


tm = len([i for i in mustard if i < 3])
tsp = len([i for i in soup if i < 3])
tsg = len([i for i in sugar if i < 3])

print(tm/len(mustard)*100)
print(tsp/len(soup)*100)
print(tsg/len(sugar)*100)
	
pt.figure('ADD')
pt.clf()
pt.plot(x_m,y_m,label='ADD',color='r')
pt.plot(x_sp,y_sp,label='ADD-S',color='g')
#pt.plot(x_sg,y_sg,label='Sugar',color='b')
pt.plot(np.full(y_m.shape,3),y_m)

pt.xlabel('ADD Threshold (Centimeters)')
pt.ylabel('ADD Percentage')
pt.title('Average Threshold')
pt.legend()
pt.show()

