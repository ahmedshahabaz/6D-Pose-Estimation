import numpy as np
from cuboid import *
from detector import *
import yaml
import matplotlib.pyplot as pt
from scipy.spatial.transform import Rotation as R
import os
from mpl_toolkits.mplot3d import Axes3D
import json

from PIL import Image
from PIL import ImageDraw

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
	global g_draw
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


# Settings
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


	# For each object to detect, load network model, create PNP solver
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


def plot_3d(show=True):
	#Plot 3D Models
	folders = [f for f in os.listdir('./models/') if f != '.DS_Store']
	mod_folder = []
	for f in folders:
		mod_folder.append(os.path.join('./models/',f))

	for mod in mod_folder:
		filename = mod+"/points.xyz"
		coordinates = []
		xyz = open(filename)
		for line in xyz:
			x,y,z = line.split()
			coordinates.append([float(x), float(y), float(z)])
		xyz.close()

		fig = pt.figure()
		ax = fig.add_subplot(111, projection='3d')

		coordinates = np.array(coordinates)*100
		print(coordinates.shape)

		ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2])
		ax.set_xlim3d(-10, 10)
		ax.set_ylim3d(-10, 10)
		ax.set_zlim3d(-10, 10)
		pt.title(mod)
	
	if show:
		pt.show()


def webcam_frames(models,pnp_solvers,pub_dimension,draw_colors):
	global g_draw
	start_time = time.perf_counter()
#Evaluate frames taken from webcam, no ground truth, just test
	files = os.listdir('./frames/')
	print(files)

	for name in files[5:7]:
		#Load image
		img = cv2.imread('./frames/'+name)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		img_copy = img.copy()
		im = Image.fromarray(img_copy)
		g_draw = ImageDraw.Draw(im)
		
		for m in models:
			# Detect object
			results = ObjectDetector.detect_object_in_image(
				models[m].net,
				pnp_solvers[m],
				img,
				config_detect
			)
			if results:
				print('Image=',name)
				print('name =',results[0]['name'])
				print('location =',results[0]['location'])
				print('quaternion =',results[0]['quaternion'])
				quat = results[0]['quaternion']
				if quat is None:
					continue
				else:
					rot = R.from_quat(results[0]['quaternion'])
					print('rotation matrix =',rot.as_dcm())
				print('cuboid2d =',results[0]['cuboid2d'])
				
			# Overlay cube on image
			for i_r, result in enumerate(results):
				if result["location"] is None:
					continue
				loc = result["location"]
				ori = result["quaternion"]

				# Draw the cube
				if None not in result['projected_points']:
					points2d = []
					for pair in result['projected_points']:
						points2d.append(tuple(pair))
					DrawCube(points2d, draw_colors[m])

		print(time.perf_counter() - start_time)
		open_cv_image = np.array(im)
		print(np.shape(open_cv_image))
		open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

		cv2.imshow('Open_cv_image', open_cv_image)
		cv2.waitKey(1)

def test_image(name,models,pnp_solvers,pub_dimension,draw_colors):
	global g_draw
	start_time = time.perf_counter()
	#Load image

	
	img = cv2.imread(name)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	img_copy = img.copy()
	im = Image.fromarray(img_copy)
	g_draw = ImageDraw.Draw(im)
	loc = []
	ori = []
	
	for m in models:
		# Detect object
		results = ObjectDetector.detect_object_in_image(
			models[m].net,
			pnp_solvers[m],
			img,
			config_detect
		)
	
				
		# Overlay cube on image
		for i_r, result in enumerate(results):
			if result["location"] is None:
				continue
			loc = np.array(result["location"])
			ori = np.array(result["quaternion"])

			# Draw the cube
			if None not in result['projected_points']:
				points2d = []
				for pair in result['projected_points']:
					points2d.append(tuple(pair))
				DrawCube(points2d, draw_colors[m])


	print(time.perf_counter() - start_time)
	return loc,ori
	#open_cv_image = np.array(im)
	#print(np.shape(open_cv_image))
	#open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

	#cv2.imshow('Open_cv_image', open_cv_image)
	#cv2.waitKey(1)


# This function returns the refined predicted point cloud of the model
def ICP(p_gt,x_n,R_real,t_real):

	# x_n is 
	#p_gt is the ground truth transformed into camera coordinate system.
	T, R_icp, t_icp = icp(x_n.transpose(),p_gt.transpose(),R_real,t_real)

	p_icp = np.zeros(p_gt.shape)

	for i in range(x.shape[1]):
		#p_gt[:,i] = np.matmul(R_gt,x[:,i]) + t_gt
		
		# R_icp and t_icp are the refined pose 
		p_icp[:,i] = np.matmul(R_icp,x_n[:,i]) + t_icp


    #p_icp is our new p_real
    
	return p_icp


if __name__ == "__main__":
	#webcam_frames(models,pnp_solvers,pub_dimension,draw_colors)
	#plot_3d()

	t_real, q_real = test_image('000030.png',models,pnp_solvers,pub_dimension,draw_colors)

	filename = "./models/0004_sugar_box/points.xyz"
	coordinates = []
	xyz = open(filename)
	for line in xyz:
		x,y,z = line.split()
		coordinates.append([float(x), float(y), float(z)])
	xyz.close()

	fig = pt.figure()
	ax = fig.add_subplot(111, projection='3d')

	coordinates = np.array(coordinates)*100
	
	#Original Model x,y,z, convert to right-handed
	x = coordinates[:,0]
	y = coordinates[:,1]
	z = -coordinates[:,2]
	
	#Plot new model coordinates where y is the height
	ax.scatter(x,z,y,'b')
	#ax.scatter(x,y,z,'r')
	ax.set_xlim3d(-10, 10)
	ax.set_ylim3d(-10, 10)
	ax.set_zlim3d(-10, 10)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	
	new_coord = np.zeros(coordinates.shape)
	new_coord[:,0] = x
	new_coord[:,1] = z
	new_coord[:,2] = y


	#folders = [f for f in os.listdir('./models/') if f == '006_mustard_bottle']
	#print(folders)

	#t_gt = np.array([115.10576923433375, -41.07723959468405, 755.6857413095329])*1/10
	#R_gt = np.array([[-0.9586628303517895, -0.2835287329246605, -0.024014888195919924], [-0.08379337156245308, 0.3619577494350433, -0.9284211691506922], [0.27192578395746053, -0.8880300419285506, -0.37075363826953395]])
	#t_real = np.array([12.062826262402817, -4.628966746830332, 77.12845553535944])
	#q_real = np.array([0.05628974, -0.91600935, -0.17087867,  0.35855099])
	#print("######################",t_real.shape,q_real.shape)
    
	with open('./test/000058/scene_gt.json') as f:
		gt = json.load(f)
	f.close()
	for x in gt['30']:
		if x['obj_id']==3:
			t_gt = np.array([x['cam_t_m2c']])*1/10
			R_gt = np.array([x['cam_R_m2c']]).reshape(3,3)

#	t_gt = np.array([115.04379958254798, -19.484092543307888, 833.3464049255888])*1/10
#	R_gt = np.array([[-0.5009382514516819, -0.8650670065754399, -0.02682610070780352], [-0.3782328793731128, 0.2466936922926899, -0.8922347901709674], [0.778460332723261, -0.4368080260234412, -0.4507750505503472]])
#	t_real = np.array([11.894122192731414, -2.090778270476108, 85.06706171906116])
#	q_real = np.array([0.1321866,-0.70362094, -0.17789202, 0.67512865])

	rot = R.from_quat(q_real)
	R_real = rot.as_dcm()
	rot = R.from_dcm(R_gt)
	q_gt = rot.as_quat()



	x = np.transpose(coordinates)
	x_n = np.transpose(new_coord)

	p_gt = np.zeros(x.shape)
	p_real = np.zeros(x.shape)


	for i in range(x.shape[1]):
		p_gt[:,i] = np.matmul(R_gt,x[:,i]) + t_gt

		# the next line is not necessary if we call the ICP function
		p_real[:,i] = np.matmul(R_real,x_n[:,i]) + t_real

	p_real = ICP(p_gt,x_n,R_real,t_real)

	#gives the minimum distance for each pair of points
	#the following lines are for calulating ADD-S. ADD-S is like single shot ICP so instead ICP can be called with max_iteration=1
	#dist, indices = correspondences(p_real.T,p_gt.T)
	#for i in range(x.shape[0]):
	#	p_real[:,i] = np.matmul(R_real,x_n[:,indices[i]]) + t_real




	
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

	dist = np.linalg.norm(p_gt-p_real,axis=0)
	print(np.mean(dist))

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