import json
import numpy as np

with open('scene_gt.json') as f:
    gt = json.load(f)
f.close()
for x in gt['1874']:
    if x['obj_id']==5:
        R_gt = np.array([x['cam_R_m2c']]).reshape((3,3))
        print(R_gt)
