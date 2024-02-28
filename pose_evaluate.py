import numpy as np
from cuboid import *
from detector import *
import yaml
from scipy.spatial.transform import Rotation as R
#import pyrealsense2 as rs
import os
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from PIL import ImageDraw
import glob
import matplotlib.pyplot as pt
from icp_test import *
from scipy.spatial.distance import cdist
from scipy.spatial import distance
from icp_test import *


### Code to visualize the neural network output

def DrawLine(point1, point2, lineColor, lineWidth):
	'''Draws line on image'''
	global g_draw
	if not point1 is None and point2 is not None:
		g_draw.line([point1, point2], fill=lineColor, width=lineWidth)


def DrawDot(point, pointColor, pointRadius):
	'''Draws dot (filled circle) on image'''
	global g_draw
	if point is not None:
		xy = [
			point[0] - pointRadius,
			point[1] - pointRadius,
			point[0] + pointRadius,
			point[1] + pointRadius
		]
		g_draw.ellipse(xy,
					   fill=pointColor,
					   outline=pointColor
					   )



def DrawCube(points, color=(255, 0, 0)):
	'''
	Draws cube with a thick solid line across
	the front top edge and an X on the top face.
	'''

	lineWidthForDrawing = 2

	# draw front
	DrawLine(points[0], points[1], color, lineWidthForDrawing)
	DrawLine(points[1], points[2], color, lineWidthForDrawing)
	DrawLine(points[3], points[2], color, lineWidthForDrawing)
	DrawLine(points[3], points[0], color, lineWidthForDrawing)

	# draw back
	DrawLine(points[4], points[5], color, lineWidthForDrawing)
	DrawLine(points[6], points[5], color, lineWidthForDrawing)
	DrawLine(points[6], points[7], color, lineWidthForDrawing)
	DrawLine(points[4], points[7], color, lineWidthForDrawing)

	# draw sides
	DrawLine(points[0], points[4], color, lineWidthForDrawing)
	DrawLine(points[7], points[3], color, lineWidthForDrawing)
	DrawLine(points[5], points[1], color, lineWidthForDrawing)
	DrawLine(points[2], points[6], color, lineWidthForDrawing)

	# draw dots
	DrawDot(points[0], pointColor=color, pointRadius=4)
	DrawDot(points[1], pointColor=color, pointRadius=4)

	# draw x on the top
	DrawLine(points[0], points[5], color, lineWidthForDrawing)
	DrawLine(points[1], points[4], color, lineWidthForDrawing)


def draw_pointcloud(p_gt,p_real):

	"""
	Input:
	*** This function can draw the point cloud comparison between ground truth and estimated point clouds for one frame

		p_gt is the ground truth point cloud in camera coordinate system
		p_gt is the estimated point cloud by the model in camera coordinate system
	"""

	fig = pt.figure()
	ax = fig.add_subplot(111, projection='3d')

	#pt.clf()
	ax.scatter(p_gt[0,:],p_gt[1,:],p_gt[2,:],'b.',s=0.5,alpha=1,label='ground truth')
	ax.scatter(p_real[0,:],p_real[1,:],p_real[2,:],'g.',s=0.5,alpha=1,label='model')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	#pt.rc('legend',fontsize=10)
	ax.legend()

	#dist = np.linalg.norm(p_gt-p_real,axis=0)
	#print(np.mean(dist))

	pt.figure()
	pt.hist(dist)
	
	x = np.linspace(0,7,200)
	#print(x.shape)
	y = np.zeros(x.shape)
	for j in range(len(x)):
		y[j] = len([i for i in dist if i < x[j]])/len(dist)
	
	pt.figure()
	pt.plot(x,y,color='g')
	pt.plot(np.full(y.shape,3),y,color='r')
	pt.xlabel('ADD Threshold (Centimeters)')
	pt.ylabel('ADD Percentage')
	pt.title('Average Threshold')
	pt.legend()
	pt.show()

def draw_add(ADD_List):
	
	"""
	*** Draws the ADD or ADD-S metric graph
	*** This function can be used to draw the graph for one frame or multiple frames
	Input:
		ADD_List: The average ADD for all the frames of one video sequence
	"""
	x = np.linspace(0,9,len(ADD_List))
	y = np.zeros(x.shape)
	for j in range(len(ADD_List)):
		y[j] = len([i for i in ADD_List if i < x[j]])/len(ADD_List)

	pt.figure()
	pt.plot(x,y)
	pt.xlabel('ADD Threshold')
	pt.ylabel('ADD Percentage')
	pt.title('Average Threshold for {}'.format(filename))
	pt.show()

def ICP(p_gt,x_n,R_real,t_real):
	"""
	Input:
		p_gt: ground truth point cloud in camera coordinate system
		x_n: estimated point cloud in model coordinate system
		R_rel: estimated Rotation matrix by the model
		t_real: estimated Translation matrix by the model

	Return:
		p_icp: refined point cloud in camera coordinate systm
	"""
	"""
	*** The returned values of icp are

		R_icp: Refined Rotation matrix
		t_icp: Refined Translation matrix
	"""
	_, R_icp, t_icp = icp(x_n.transpose(),p_gt.transpose(),R_real,t_real)

	p_icp = np.zeros(p_gt.shape)

	for i in range(x.shape[1]):
		#p_gt[:,i] = np.matmul(R_gt,x[:,i]) + t_gt
		
		# R_icp and t_icp are the refined pose 
		p_icp[:,i] = np.matmul(R_icp,x_n[:,i]) + t_icp


   	#p_icp is our new p_real
    
	return p_icp



def evaluate(filename,t_gt,R_gt,t_real,q_real,eval_metric = "ADD"):
	"""
	: Param filename: directory to the file that has the ground truth points
	: Param t_gt: ground truth translation matrix
	: Param R_gt: ground truth Rotation matrix
	: Param t_real: predicted/estimated translation matrix by the model
	: Param q_real: preditected/estimated Quarternion (rotation)

	:return : Returns the calculated Average Distance (ADD/ADD-S) for a single frame

	Function to evaluate the model
	"""

	coordinates = []
	xyz = open(filename)
	for line in xyz:
		x,y,z = line.split()
		coordinates.append([float(x), float(y), float(z)])
	xyz.close()

	#fig = pt.figure()
	#ax = fig.add_subplot(111, projection='3d')

	coordinates = np.array(coordinates)*100
	
	#Original Model x,y,z, convert to right-handed
	x = coordinates[:,0]
	y = coordinates[:,1]
	z = -coordinates[:,2]
	
	
	new_coord = np.zeros(coordinates.shape)
	new_coord[:,0] = x
	new_coord[:,1] = z
	new_coord[:,2] = y

	#Trasnforming the Quarternion to 3D Rotation Matrix
	rot = R.from_quat(q_real)
	R_real = rot.as_dcm()
	rot = R.from_dcm(R_gt)
	q_gt = rot.as_quat()

	x = np.transpose(coordinates)
	x_n = np.transpose(new_coord)

	p_gt = np.zeros(x.shape)
	p_real = np.zeros(x.shape)

	## Regress the estimation to ground truth by default
	for i in range(x.shape[1]):
		p_gt[:,i] = np.matmul(R_gt,x[:,i]) + t_gt
		p_real[:,i] = np.matmul(R_real,x_n[:,i]) + t_real

	## Do the evaluation based on the ADD-metric
	#print(x_n.shape)
	if (eval_metric.lower() == 'add'):
		dist = np.linalg.norm(p_gt-p_real,axis=0)

	elif (eval_metric.lower() == 'add-s'):
		#calculating the all pair distances
		distances = cdist(p_gt.T, p_real.T, metric = 'euclidean')
		#calculating the pair of points with minimum distance
		dist = np.amin(distances, axis = 1)
		#gives the indices of the minimum distance
		indices = np.argmin(distances, axis = 1)
		# now regress the estimation to the corresponding lowest distance pair
		for i in range(x.shape[0]):
			p_real[:,i] = np.matmul(R_real,x_n.T[indices[i]].T) + t_real

	elif (eval_metric.lower() == 'icp'):
		p_real = ICP(p_gt,x_n,R_real,t_real)
		dist = np.linalg.norm(p_gt-p_real,axis=0)

	# to draw the point cloud comparison	
	elif (eval_metric.lower() == 'point cloud'):
		draw_pointcloud(p_gt,p_real)

	
    
	ADD = np.mean(dist)
	print(eval_metric.upper(),ADD)	

	return ADD

# Settings
#config_name = "my_config_webcam.yaml"
#config_name = "my_config_realsense.yaml"

# Load the config files in the config file the we have to specify the 
# Object model (the objects we want to detect the pose for) for which the will be loaded

config_name = "my_config_test.yaml"
exposure_val = 166


yaml_path = 'cfg/{}'.format(config_name)
with open(yaml_path, 'r') as stream:
	try:
		print("Loading DOPE parameters from '{}'...".format(yaml_path))
		params = yaml.load(stream)
		print('    Parameters loaded.')
	except yaml.YAMLError as exc:
		print(exc)


	models = {}
	pnp_solvers = {}
	pub_dimension = {}
	draw_colors = {}

	# Initialize parameters
	matrix_camera = np.zeros((3,3))
	matrix_camera[0,0] = params["camera_settings"]['fx']
	matrix_camera[1,1] = params["camera_settings"]['fy']
	matrix_camera[0,2] = params["camera_settings"]['cx']
	matrix_camera[1,2] = params["camera_settings"]['cy']
	matrix_camera[2,2] = 1
	dist_coeffs = np.zeros((4,1))

	if "dist_coeffs" in params["camera_settings"]:
		dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
	config_detect = lambda: None
	config_detect.mask_edges = 1
	config_detect.mask_faces = 1
	config_detect.vertex = 1
	config_detect.threshold = 0.5
	config_detect.softmax = 1000
	config_detect.thresh_angle = params['thresh_angle']
	config_detect.thresh_map = params['thresh_map']
	config_detect.sigma = params['sigma']
	config_detect.thresh_points = params["thresh_points"]


	# For each object to detect, load network model, create PNP solver, and start ROS publishers
	for model in params['weights']:
		models[model] = \
			ModelData(
				model,
				"weights/" + params['weights'][model]
			)
		models[model].load_net_model()

		draw_colors[model] = tuple(params["draw_colors"][model])

		pnp_solvers[model] = \
			CuboidPNPSolver(
				model,
				matrix_camera,
				Cuboid3d(params['dimensions'][model]),
				dist_coeffs=dist_coeffs
			)

# RealSense Start
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# profile = pipeline.start(config)
# # Setting exposure
# s = profile.get_device().query_sensors()[1]
# s.set_option(rs.option.exposure, exposure_val)

#cap = cv2.VideoCapture(0)



start_time = time.perf_counter()

#idx = 0

#""
#         /\_/\
#    ____/ o o \
#  /~____  =ø= /
#(______)__m_m)
#------------------------------------------------
#Thank you for visiting https://asciiart.website/
#This ASCII pic can be found at
#https://asciiart.website/index.php?art=animals/cats

#manual settings

"""
obj_id = 2 for cracker box
obj_id = 3 for sugar box
obj_id = 4 for tomato soup can
obj_id = 5 for mustard bottle
"""
obj_id = 3

# List of the folders of YCB-V Test dataset for which we want to Evaluate
# Each folder has the frames of a video sequence
folder_list = ['49','51','54','55','58']

# Folder containing the ground truth points in model coordinate system for the object we want to Evaluate

if obj_id == 2:
	filename = "./models/003_cracker_box/points.xyz"
elif obj_id == 3:
	filename = "./models/004_sugar_box/points.xyz"
elif obj_id == 4:
	filename = "./models/005_tomato_soup_can/points.xyz"
elif obj_id == 5:
	filename = "./models/006_mustard_bottle/points.xyz"

# Stores all the ADDs for each frame containing object with id = obj_id
ADD_model = []

for folder in folder_list:

    # the folder from which we want to load all the frame
    parent_dir = './test/0000'+folder+'/'
    
    rgb_folder = os.listdir(parent_dir+'rgb/')
    frame_name = [str(int(i[:-4])) for i in rgb_folder]
    
    # loading the json file with the ground truths
    with open(parent_dir+'scene_gt.json') as f:
        gt = json.load(f)
    f.close()

    # Counts the number of frames processed in a folder 
    count = 0

    print(parent_dir)

    # Stores all the evaluated results for all the frames in a folder
    ADD_List = []
   
    for file,frame in zip(rgb_folder,frame_name):

        count +=1 
        img = cv2.imread(parent_dir+'rgb/'+file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)	
        g_draw = ImageDraw.Draw(im)

        # load the model to detect the objects
        # The model has to be loaded multiple times for detecting multiple objects
        for m in models:
            results = ObjectDetector.detect_object_in_image(
				models[m].net,
				pnp_solvers[m],
				img,
				config_detect
            )

            """
            t_real = Translation matrix estimated by the model
            q_real = Quarternion (Rotation) estimated by the model

            t_gt = ground truth Translation matrix
            R_gt = ground truth Rotation matrix
            """

            for i_r,result in enumerate(results):
                if result['location'] is None:
                    continue
                t_real = np.array(result['location'])
                q_real = np.array(result['quaternion'])
                print(i_r,'location =',t_real)
                print('quaternion =',q_real)
				#break
            
            for x in gt[frame]:	
                if x['obj_id']==obj_id:	
                    t_gt = np.array([x['cam_t_m2c']])*1/10
                    R_gt = np.array([x['cam_R_m2c']]).reshape(3,3)
                    print('found')

            # Call the evalute function to evalute the result of the model 
            # Append the return value to ADD_List
            # Default evaluation metric is ADD
            ADD_List.append(evaluate(filename,t_gt,R_gt,t_real,q_real))

            # Pass the parameter eval_metric = 'ADD-S' for evaluating with ADD-S metric
            #ADD_List.append(evaluate(filename,t_gt,R_gt,t_real,q_real, eval_metric = 'ADD-S'))
            
        print("#", count,' Elapsed Time=',time.perf_counter() - start_time)
        print("__________________________________________________________________")
		
	
		# Overlay cube on image
		#for i_r, result in enumerate(results):
		#	if result["location"] is None:
		#		continue
			#print("***********************************************")
			#print("projected_points : ",result["projected_points"])
            
			# Draw the cube
			#if None not in result['projected_points']:
			#	points2d = []
			#	for pair in result['projected_points']:
			#		points2d.append(tuple(pair))
			#	DrawCube(points2d, draw_colors[m])

	
	#open_cv_image = np.array(im)
	#print(np.shape(open_cv_image))
	#open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)



    	

#pt.figure()
#pt.imshow(open_cv_image)
#pt.show()


    print(ADD_List)
    # Saving the ADD calculated for all the frames in a folder to a .npy file
    np.save(parent_dir[7:13]+'_' +str(obj_id) +'.npy',np.array(ADD_List))
    print(parent_dir[7:13]+'_' +str(obj_id)+'.npy')
    ADD_List = np.load(parent_dir[7:13]+'_' +str(obj_id) +'.npy')
    ADD_model.append(ADD_List)
    #draw_add(ADD_List)

draw_add(ADD_model)
np.save(filename[10,:],np.array(ADD_model))
    




