import numpy as np 
from scipy.io import loadmat,savemat
from PIL import Image
from core.skin import skinmask
import argparse
from Flame.utils.util import *
import os
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#calculating least square problem
def POS(xp,x):
	npts = xp.shape[1]

	A = np.zeros([2*npts,8])

	A[0:2*npts-1:2,0:3] = x.transpose()
	A[0:2*npts-1:2,3] = 1

	A[1:2*npts:2,4:7] = x.transpose()
	A[1:2*npts:2,7] = 1;

	b = np.reshape(xp.transpose(),[2*npts,1])

	k,_,_,_ = np.linalg.lstsq(A,b)

	R1 = k[0:3]
	R2 = k[4:7]
	sTx = k[3]
	sTy = k[7]
	s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
	t = np.stack([sTx,sTy],axis = 0)

	return t,s

# resize and crop images
def resize_n_crop_img(img,lm,t,s,target_size):
	w0,h0 = img.size
	w = (w0/s*102).astype(np.int32)
	h = (h0/s*102).astype(np.int32)
	img = img.resize((w,h),resample = Image.BICUBIC)

	left = (w/2 - target_size/2 + float((t[0] - w0/2)*102/s)).astype(np.int32)
	right = left + target_size
	up = (h/2 - target_size/2 + float((h0/2 - t[1])*102/s)).astype(np.int32)
	below = up + target_size

	img = img.crop((left,up,right,below))
	img = np.array(img)
	img = img[:,:,::-1] #RGBtoBGR
	img = np.expand_dims(img,0)
	lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102
	lm = lm - np.reshape(np.array([(w/2 - target_size/2),(h/2-target_size/2)]),[1,2])

	return img,lm


# resize and crop input images before sending to the R-Net
def align_img(img,lm,lm3D, target_size):

	w0,h0 = img.size

	# change from image plane coordinates to 3D sapce coordinates(X-Y plane)
	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)
	lm = get5(lm)


	# calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
	t,s = POS(lm.transpose(),lm3D.transpose())

	# processing the image
	img_new,lm_new = resize_n_crop_img(img,lm,t,s, target_size)
	lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
	trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])

	return img_new,lm_new,trans_params

# detect 68 face landmarks for aligned images
def get_68landmark(img,detector,sess):

	input_img = detector.get_tensor_by_name('input_imgs:0')
	lm = detector.get_tensor_by_name('landmark:0')

	landmark = sess.run(lm,feed_dict={input_img:img})
	landmark = np.reshape(landmark,[68,2])
	landmark = np.stack([landmark[:,1],223-landmark[:,0]],axis=1)

	return landmark

# get skin attention mask for aligned images
def get_skinmask(img):

	#img = np.squeeze(img,0)
	skin_img = skinmask(img)
	return skin_img

def parse_args():
    desc = "Data preprocessing for Deep3DRecon."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--img_path', type=str, default='./input', help='original images folder')
    parser.add_argument('--save_path', type=str, default='./processed_data', help='custom path to save proccessed images and labels')


    return parser.parse_args()

def get5(lms68):
	lm_idx = np.array([31,37,40,43,46,49,55]) - 1
	lms68 = np.stack([lms68[lm_idx[0],:],np.mean(lms68[lm_idx[[1,2]],:],0),np.mean(lms68[lm_idx[[3,4]],:],0),lms68[lm_idx[5],:],lms68[lm_idx[6],:]], axis = 0)
	return lms68[[1,2,0,3,4],:]

# training data pre-processing
def preprocess(img, model, lm3D, img_lm, target_size):


	model5 = get5(lm3D)
	

	img_align,_,_ = align_img(img,img_lm,model5, target_size)  # [1,224,224,3] BGR image

	skin_mask = get_skinmask(img_align)

	img_align.squeeze(0)[:,:,::-1].astype(np.uint8)
	skin_mask.astype(np.uint8)

	return img_align, skin_mask

			
