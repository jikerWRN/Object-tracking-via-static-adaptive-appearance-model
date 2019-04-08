#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pylab
import os
import cv2
import STSAM


pylab.mpl.rcParams['font.sans-serif'] = ['SimHei'] 
pylab.mpl.rcParams['axes.unicode_minus'] = False 

def read_boxes_from_txt(path):
	with open(path, 'r') as f:
		content = [line.strip() for line in f.readlines()]
	if content[0].find(',')!=-1: 
		rows = [line.strip().split(',') for line in content]
	elif content[0].find('\t')!=-1:
		rows = [line.strip().split('\t') for line in content]
	elif content[0].find('   ')!=-1:
		rows = [line.strip().split('   ') for line in content]
	return rows

def cut_the_white_side(dir, file_list):
	for file in file_list:
		path = dir + file + '.png'
		src = cv2.imread(path)
		row_white = src[:,0,:]
		# print row_white.shape
		col_white = src[0,:,:]
		# print col_white.shape
		for r in range(0,src.shape[1]):
			if np.any(row_white-src[:,r,:]):
				r_start_index = r
				break
		for r in range(src.shape[1]-1,0,-1):
			if np.any(row_white-src[:,r,:]):
				r_end_index = r
				break
		for c in range(0,src.shape[0]):
			if np.any(col_white-src[c,:,:]):
				c_start_index = c
				break
		for c in range(src.shape[0]-1,0,-1):
			if np.any(col_white-src[c,:,:]):
				c_end_index = c
				break
		cv2.imwrite(dir+'cut/'+file+'.png', src[c_start_index:c_end_index+1,
										r_start_index:r_end_index+1,
										:])

def run_tracker(estimate_scale, estimate_rotation, input_box, input_pic, output):
	STSAM.main(estimate_scale, estimate_rotation, input_box, input_pic, output)

if __name__ == '__main__':
	input_box = "198,214,34,81"
	#input_pic = "../vot2013/jump/{:08d}.jpg"
	input_pic="E:/gp/sequence/Basketball/img/{:04d}.jpg"
	#input_pic = ur"E:/gp/sequence/UAV123/data_seq/UAV123/person23/img/{:06d}.jpg"
	#input_pic = ur"E:/gp/sequence/pafiss_eval_dataset/sequence09/img/img{:06d}.jpeg"
	#input_pic = ur"G:\gp\卫老师\师兄工作交接\代码\vot2013\david\{:08d}.jpg".encode('GBK')
	# output = '../vot2013/jump_output'
	output="E:/gp/sequence/Basketball/output"
	#output = ur"E:/gp/sequence/UAV123/data_seq/UAV123/person23/img_output1"
	#output = ur"E:/gp/sequence/pafiss_eval_dataset/sequence09/img_output1"
	#output = u"G:\gp\卫老师\师兄工作交接\代码\vot2013\bicycle_output".encode('GBK')
	run_tracker(True, True,input_box, input_pic, output)
	# print read_boxes_from_txt(u"./ttt.txt")
	#dir = './'
	#filelist = ['001']
	#cut_the_white_side(dir, filelist)
