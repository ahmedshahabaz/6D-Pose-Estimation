import json
import numpy as np
import cv2 
import glob
filenames = glob.glob("rgb/*.png")

filenames.sort()

with open('scene_gt.json') as f:
    gt = json.load(f)
f.close()

for file in filenames:
    start = 6
    print(file[6])
    while((file[start])=='0'):
        start+=1
        print(start)
    print(file[start:10])
    break
    






#images = [cv2.imread(img) for img in filenames]
#i = 0
#images = []
##print(filenames)
#
#for f in filenames:
#    print(f)
#    img = cv2.imread(f)
#    images.append(img)
#    
##print(images[74].shape)
#name = filenames[71][6:10]
#print(filenames[71][4:])
#
#with open('scene_gt.json') as f:
#    gt = json.load(f)
#f.close()
#for x in gt[name]:
#    if x['obj_id']==5:
#        R_gt = np.array([x['cam_R_m2c']]).reshape((3,3))
#        print(R_gt)

#img = images[71]
#open_cv_image = np.array(img)
#print(np.shape(open_cv_image))
##open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
#cv2.imshow('Open_cv_image', open_cv_image)
#cv2.waitKey(0)
#cv2.destroyallwindows()        
#cv2.imshow('Open_cv_image', images[1])
#

#for img in images:
#    i+=1
#    cv2.imshow(img,i)
#    if i == 3:
#        break