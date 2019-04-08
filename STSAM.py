#coding=utf-8

'''
STSAM = Self-correction Object Tracking Algorithm Based on Static-Adaptive Appearance Model
'''
import argparse
import cv2
from numpy import empty, nan
import os
import sys
import time

import numpy as np
import mosse
import CMT
import util

parser = argparse.ArgumentParser(description='Self-correction Object Tracking Algorithm Based on Static-Adaptive Appearance Model')
parser.add_argument('inputpath', nargs='?', help='The input path.')   
parser.add_argument('--bbox', dest='bbox', help='Specify initial bounding box.')
parser.add_argument('--output-dir', dest='output', help='Specify a directory for output data.')
args = parser.parse_args()
CMT = CMT.CMT()


def main(estimate_scale = True,
		estimate_rotation = True,
		input_box = None,
		input_pic = None,
		output = None):
	CMT.estimate_scale = estimate_scale
	CMT.estimate_rotation = estimate_rotation 
	pause_time = 10 
	skip =None 
	preview = None
	quiet = False

	if input_pic is not None:
		print input_pic
		input_dir = input_pic
	elif args.inputpath is not None:
		print args.inputpath
		input_dir = args.inputpath
	else:
		input_dir = None

	if output is not None:
		output_dir = output
	elif args.output is not None:
		output_dir = args.output
	else:
		output_dir = None

	if input_box is not None:
		init_box = input_box
	elif args.bbox is not None:
		init_box = args.bbox
	else:
		init_box = None

	if output_dir is not None:
		quiet = True
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)
		elif not os.path.isdir(output_dir):
			raise Exception(output_dir + ' exists, but is not a directory')

	# Clean up
	cv2.destroyAllWindows()

	if input_dir is not None:

		# If a path to a file was given, assume it is a single video file
		if os.path.isfile(input_dir):
			cap = cv2.VideoCapture(input_dir)

			#Skip first frames if required
			if skip is not None:
				cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, skip)     


		# Otherwise assume it is a format string for reading images
		else:
			cap = util.FileVideoCapture(input_dir)

			#Skip first frames if required
			if skip is not None:
				cap.frame = 1 + skip

		# By default do not show preview in both cases
		if preview is None:
			preview = False

	else:
		# If no input path was specified, open camera device
		cap = cv2.VideoCapture(0)
		if preview is None:
			preview = True

	# Check if videocapture is working
	if not cap.isOpened():
		print 'Unable to open video input.'
		sys.exit(1)

	while preview:
		status, im = cap.read()
		cv2.imshow('Preview', im)
		k = cv2.waitKey(10)
		if not k == -1:
			break

	# Read first frame
	status, im0 = cap.read()
	im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
	im_draw = np.copy(im0)

	if init_box is not None:
		# Try to disassemble user specified bounding box
		values = init_box.split(',')    
		try:
			values = [int(v) for v in values]       
			                                        
		except:
			raise Exception('Unable to parse bounding box')
		if len(values) != 4:
			raise Exception('Bounding box must have exactly 4 elements')
		bbox = np.array(values)     

		# Convert to point representation, adding singleton dimension
		bbox = util.bb2pts(bbox[None, :])      #bbox[None, :]=array([[a, b, c,d]])

		# Squeeze
		bbox = bbox[0, :]

		tl = bbox[:2]
		br = bbox[2:4]
		
		
	else:
		# Get rectangle input from user
		(tl, br) = util.get_rect(im_draw)

	print 'using', tl, br, 'as init bb'


	#CMT.initialise(im0, tl, br)
	CMT.initialise(im_gray0, tl, br)
	#util.draw_keypoints(selected_keypoints, im_gray0,(255, 0, 0)) 

	frame = 1
	while True:
		# Read image
		status, im = cap.read()
		if not status:
			break
		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im_draw = np.copy(im)

		tic = time.time()
		CMT.process_frame(im)
		#CMT.process_frame(im_gray)
		toc = time.time()

		# Display results

		# Draw updated estimate
		if CMT.has_result:

			cv2.line(im_draw, CMT.tl, CMT.tr, (255, 0, 0), 2)
			cv2.line(im_draw, CMT.tr, CMT.br, (255, 0, 0), 2)
			cv2.line(im_draw, CMT.br, CMT.bl, (255, 0, 0), 2)
			cv2.line(im_draw, CMT.bl, CMT.tl, (255, 0, 0), 2)
			# cv2.imshow("im_draw1",im_draw)
			# cv2.waitKey()

		#util.draw_keypoints(CMT.tracked_keypoints, im_draw, (255, 255, 255))
		# this is from simplescale
		#util.draw_keypoints(CMT.votes[:, :2], im_draw)  # blue
		#util.draw_keypoints(CMT.outliers[:, :2], im_draw, (0, 0, 255))

		if output_dir is not None:
			
			# Original image
			# cv2.imwrite('{0}/input_{1:08d}.png'.format(output_dir, frame), im)
			# Output image
			cv2.imwrite('{0}/output_{1:08d}.png'.format(output_dir, frame), im_draw)
			'''
			# Keypoints
			with open('{0}/keypoints_{1:08d}.csv'.format(output_dir, frame), 'w') as f:
				f.write('x y\n')
				np.savetxt(f, CMT.tracked_keypoints[:, :2], fmt='%.2f')

			# Outlier
			with open('{0}/outliers_{1:08d}.csv'.format(output_dir, frame), 'w') as f:
				f.write('x y\n')
				np.savetxt(f, CMT.outliers, fmt='%.2f')

			# Votes
			with open('{0}/votes_{1:08d}.csv'.format(output_dir, frame), 'w') as f:
				f.write('x y\n')
				np.savetxt(f, CMT.votes, fmt='%.2f')
			'''
			# Bounding box
			# with open('{0}/bbox_{1:08d}.csv'.format(output_dir, frame), 'w') as f:
			with open('{0}/bbox.txt'.format(output_dir, frame), 'a') as f:
				# f.write('x y\n')
				# Duplicate entry tl is not a mistake, as it is used as a drawing instruction
				# np.savetxt(f, np.array((CMT.tl, CMT.tr, CMT.br, CMT.bl, CMT.tl)), fmt='%.2f')
				L = []
				for i in [CMT.tl, CMT.tr, CMT.br, CMT.bl]:
					#st = '(' + str(int(i[0])) + ' ' + str(int(i[1])) + ')'
					st = '(' + str(float(i[0])) + ' ' + str(float(i[1])) + ')'
					L.append(st)
				bb = ','.join(L)
				f.write(bb + '\n')

		if not quiet:
			cv2.imshow('main', im_draw)

			# Check key input
			k = cv2.waitKey(pause_time)
			key = chr(k & 255)
			if key == 'q':
				break
			if key == 'd':
				import ipdb; ipdb.set_trace()

		# Remember image
		im_prev = im_gray

		# Advance frame number
		frame += 1

		# print '{5:04d}: center: {0:.2f},{1:.2f} scale: {2:.2f}, active: {3:03d}, {4:04.0f}ms'.format(CMT.center[0], CMT.center[1], CMT.scale_estimate, CMT.active_keypoints.shape[0], 1000 * (toc - tic), frame)

if __name__ == '__main__':
	main()
