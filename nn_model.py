import pickle
import sys
import time
import warnings
from math import floor

import lasagne
from lasagne import init
from lasagne.init import GlorotUniform
try:
	from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
	from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
	print('Using cuda_convnet (faster)')
except ImportError:
	from lasagne.layers import Conv2DLayer as Conv2DLayer
	from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer
	print('Using lasagne.layers (slower)')
from lasagne.layers import Layer
from lasagne.layers import Conv2DLayer as Conv2DLayer
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.layers import MaxPool2DLayer as MaxPool2DLayer
from lasagne import nonlinearities
from lasagne.nonlinearities import rectify, leaky_rectify, tanh, sigmoid, softmax
from lasagne.layers.normalization import batch_norm
from lasagne.objectives import categorical_crossentropy, binary_crossentropy
from lasagne.updates import nesterov_momentum, adam

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import theano 
import theano.tensor as T

import scipy as sp
import scipy.ndimage.morphology
from scipy.misc import imresize, imread

warnings.filterwarnings('ignore') # we ignore a RuntimeWarning produced from dividing by zero
np.random.seed(1234)


def load_StereoImages(dirpath='/Users/Wojtek/Documents/Python Projects/DepthMap_dataset'):


	# load training data
	X, X_left, X_right, y = [], [], [], []
	n = 0
	for n in range(1,1000):
		path = ('{}/Depth_map/DepthMap_{}.png').format(dirpath,n)

		with open(path, 'rb') as f:

			depthM = imread(f)
			size_depthM = depthM.shape
			depthM = Image.open(f)
			depthM = depthM.convert('L')
			depthM = np.array(depthM.getdata(), dtype=np.uint8)

		y.append(depthM)

	y = (np.concatenate(y).
			reshape(-1, 1, size_depthM[0], size_depthM[1]).astype(np.float32))


	for n in range(1,1000):

		path = ('{}/StereoImages/Stereoscopic_{}.png').format(dirpath,n)
		with open(path, 'rb') as f:

			StereoIm = imread(f)
			size_StereoIm = StereoIm.shape
			StereoIm = Image.open(f)
			StereoIm = StereoIm.convert('L')
			StereoIm = (np.array(StereoIm.getdata(),dtype=np.uint8).
						reshape(1, size_StereoIm[0], size_StereoIm[1]))
			Im_left = StereoIm[0,...,0:size_StereoIm[1]//2]
			Im_right = StereoIm[0,...,size_StereoIm[1]//2:size_StereoIm[1]]

		X.append(StereoIm)
		X_left.append(Im_left)
		X_right.append(Im_right)

	X = (np.concatenate((X_left,X_right),1).
		reshape(-1, 2, size_StereoIm[0], size_StereoIm[1]//2).
		astype(np.float32))

	X_left = (np.concatenate(X_left).
				reshape(-1, 1, size_StereoIm[0], size_StereoIm[1]//2).
				astype(np.float32))

	X_right = (np.concatenate(X_right).
				reshape(-1, 1, size_StereoIm[0], size_StereoIm[1]//2).
				astype(np.float32))


	print(X_left.shape, X_right.shape, X.shape)

	ii = np.random.permutation(len(X))
	X_train = X[ii[floor(len(X)*0.1):]]
	X_left_train = X_left[ii[floor(len(X_left)*0.1):]]
	X_right_train = X_right[ii[floor(len(X_right)*0.1):]]
	y_train = y[ii[floor(len(X)*0.1):]]

	X_valid = X[ii[:floor(len(X)*0.1)]]
	X_left_valid = X_left[ii[:floor(len(X_left)*0.1)]]
	X_right_valid = X_right[ii[:floor(len(X_right)*0.1)]]
	y_valid = y[ii[:floor(len(X)*0.1)]]

	# normalize to zero mean and unity variance
	offset = np.mean(X_train, 0)
	scale = np.std(X_train, 0).clip(min=1)

	y_offset = np.mean(y_train, 0)
	y_scale = y_train.max()

	offset_left = np.mean(X_left_train, 0)
	scale_left = np.std(X_left_train, 0).clip(min=1)
	offset_right = np.mean(X_right_train, 0)
	scale_right = np.std(X_right_train, 0).clip(min=1)

	X_train = (X_train - offset) / scale
	X_valid = (X_valid - offset) / scale
	
	X_left_train = (X_left_train - offset_left) / scale_left
	X_right_train = (X_right_train - offset_right) / scale_right

	X_left_valid = (X_left_valid - offset_left) / scale_left
	X_right_valid = (X_right_valid - offset_right) / scale_right

	y_train = y_train / y_scale
	y_valid = y_valid / y_scale


	return X_train, X_left_train, X_right_train, y_train, X_valid, X_left_valid, X_right_valid, y_valid


def build_stereo_cnn(input_var=None):
	
	conv_num_filters1 = 16
	conv_num_filters2 = 32
	conv_num_filters3 = 64
	conv_num_filters4 = 128
	filter_size1 = 7
	filter_size2 = 5
	filter_size3 = 3
	filter_size4 = 3
	pool_size = 2
	scale_factor = 2
	pad_in = 'valid'
	pad_out = 'full'

	# Input layer, as usual:
	network = InputLayer(
						shape=(None, 2, X_train.shape[2], X_train.shape[3]),
						input_var=input_var,
						name="input_layer")

	network = batch_norm(Conv2DLayer(
							network,
							num_filters=conv_num_filters1,
							filter_size=(filter_size1, filter_size1),
							pad=pad_in,
							nonlinearity=lasagne.nonlinearities.rectify,
							W=lasagne.init.GlorotUniform(),
							name="conv1"))

	network = MaxPool2DLayer(
							network,
							pool_size=(pool_size, pool_size),
							name="pool1")

	network = batch_norm(Conv2DLayer(
							network,
							num_filters=conv_num_filters2,
							filter_size=(filter_size2, filter_size2),
							pad=pad_in,
							nonlinearity=lasagne.nonlinearities.rectify,
							W=lasagne.init.GlorotUniform(),
							name="conv2"))

	network = MaxPool2DLayer(
							network,
							pool_size=(pool_size, pool_size),
							name="pool2")

	network = batch_norm(Conv2DLayer(
							network,
							num_filters=conv_num_filters3,
							filter_size=(filter_size3, filter_size3),
							pad=pad_in,
							nonlinearity=lasagne.nonlinearities.rectify,
							W=lasagne.init.GlorotUniform(),
							name="conv3"))

	network = MaxPool2DLayer(
							network,
							pool_size=(pool_size, pool_size),
							name="pool3")

	network = batch_norm(Conv2DLayer(
							network,
							num_filters=conv_num_filters4,
							filter_size=(filter_size4, filter_size4),
							pad=pad_in,
							nonlinearity=lasagne.nonlinearities.rectify,
							W=lasagne.init.GlorotUniform(),
							name="conv4"))

	network = batch_norm(Conv2DLayer(
							network,
							num_filters=32,
							filter_size=(filter_size4, filter_size4),
							pad=pad_out,
							nonlinearity=lasagne.nonlinearities.rectify,
							W=lasagne.init.GlorotUniform(),
							name="deconv1"))

	network = Upscale2DLayer(
							network,
							scale_factor=(pool_size, pool_size),
							name="upscale1")

	network = batch_norm(Conv2DLayer(
							network,
							num_filters=16,
							filter_size=(filter_size3, filter_size3),
							pad=pad_out,
							nonlinearity=lasagne.nonlinearities.rectify,
							W=lasagne.init.GlorotUniform(),
							name="deconv2"))

	network = Upscale2DLayer(
							network,
							scale_factor=(pool_size, pool_size),
							name="upscale2")

	network = batch_norm(Conv2DLayer(
							network,
							num_filters=8,
							filter_size=(filter_size2, filter_size2),
							pad=pad_out,
							nonlinearity=lasagne.nonlinearities.rectify,
							W=lasagne.init.GlorotUniform(),
							name="deconv3"))

	network = Upscale2DLayer(
							network,
							scale_factor=(pool_size, pool_size),
							name="upscale3")

	network = batch_norm(Conv2DLayer(
							network,
							num_filters=1,
							filter_size=(filter_size1, filter_size1),
							pad=pad_out,
							nonlinearity=lasagne.nonlinearities.sigmoid,
							W=lasagne.init.GlorotUniform(),
							name="deconv4"))

	return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	
	assert len(inputs) == len(targets)
	
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		
		yield inputs[excerpt], targets[excerpt]


def iterator(X, batchsize):
	
	indices = np.arange(len(X))
	for i in range(0, len(X) - batchsize + 1, batchsize):
		
		sli = indices[i:i+batchsize]
		yield X[sli]


def save_params(model, fn):
	
	with open(fn, 'w') as wr:
	
		pickle.dump(lasagne.layers.get_all_param_values(model), wr)
		

def dice_loss(predictions, targets):
	
	dice_index = (2*T.sum(predictions*targets)
				  /((T.sum(predictions)
				  +T.sum(targets))))
	
	return -dice_index


def main(model='cnn', num_epochs=200):
	
	# Load the dataset
	print("Loading data...")
	X_train, X_left_train, X_right_train, y_train, X_val, X_left_val, X_right_val, y_val = load_StereoImages()

	print(type(X_train), len(X_train))
	
	indices = np.arange(len(X_train))
	np.random.shuffle(indices)
	
	y_train = y_train[...,3:y_train.shape[2]-3,:]
	y_val = y_val[...,3:y_val.shape[2]-3,:]
	
	print('X_train type and shape:', X_train.dtype, X_train.shape)
	print('X_train.min():', X_train.min())
	print('X_train.max():', X_train.max())

	print('X_val type and shape:', X_val.dtype, X_val.shape)
	print('X_val.min():', X_val.min())
	print('X_val.max():', X_val.max())
	
	print('y_train type and shape:', y_train.dtype, y_train.shape)
	print('y_train.min():', y_train.min())
	print('y_train.max():', y_train.max())

	print('y_val type and shape:', y_val.dtype, y_val.shape)
	print('y_val.min():', y_val.min())
	print('y_val.max():', y_val.max())

	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs',dtype=theano.config.floatX)
	target_var = T.tensor4('targets',dtype=theano.config.floatX)

	# Create neural network model (depending on first command line parameter)
	print("Building model and compiling functions...")

	network = build_stereo_cnn(input_var)
	laylist = lasagne.layers.get_all_layers(network)
	
	for l in laylist:
		print(l.name, lasagne.layers.get_output_shape(l))
		
	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	dloss = lasagne.objectives.squared_error(prediction, target_var)
	dloss = dloss.mean()
	
	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates2 = lasagne.updates.adam(dloss, params, learning_rate=0.01)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss2 = lasagne.objectives.squared_error(test_prediction, target_var)
	test_loss2 = test_loss2.mean()

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn2 = theano.function(
								inputs=[input_var, target_var],
								outputs=dloss,
								updates=updates2)

	# Compile a second function computing the validation loss and accuracy:
	val_fn2 = theano.function(
							  inputs=[input_var, target_var],
							  outputs=[test_loss2])

	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, y_train, 16, shuffle=True):
			inputs, targets = batch
			train_err += train_fn2(inputs, targets)
			train_batches += 1
		# And a full pass over the validation data:
		val_err = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 16, shuffle=False):
			inputs, targets = batch
			err = val_fn2(inputs, targets)
			val_err += err[0]
			val_batches += 1

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
													epoch + 1,
													num_epochs,
													time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  ratio:\t\t{:.6f}".format((train_err / train_batches)
										   /(val_err / val_batches)))
		#filename = "model4_loss:\t\t{:.6f}".format(train_err / train_batches)
		np.savez('model_stereo_v1.2.npz', *lasagne.layers.get_all_param_values(network))

	# Load model again later on like this:
	# with np.load('model.npz') as f:
	#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	# lasagne.layers.set_all_param_values(network, param_values)
	return X_train,X_left_train,X_right_train, y_train, X_val,X_left_val,X_right_val, y_val



if __name__ == "__main__":

	#X_train,X_left_train,X_right_train, y_train, X_val,X_left_val,X_right_val, y_val = main(model='cnn', num_epochs=45)

	X_train,X_left_train,X_right_train, y_train, X_valid,X_left_valid,X_right_valid, y_valid = load_StereoImages()
	print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)


	plt.imshow(y_valid[11,0,...], cmap=cm.Greys_r)
	plt.show()
	plt.imshow(X_valid[11,0,...], cmap=cm.Greys_r)
	plt.show()


	input_var = T.tensor4('inputs', dtype=theano.config.floatX)
	network = build_stereo_cnn(input_var)
	with np.load('model_stereo_v1.2.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(network, param_values)
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	pred_fn = theano.function([input_var], [test_prediction])


	n = 0
	f, axarr = plt.subplots(7, 4, sharey='col')
	f.set_figheight(25)
	f.set_figwidth(15)
	for Im_num in range(50, 57):

		left_eye = X_valid[Im_num,0,...]
		right_eye = X_valid[Im_num,1,...]

		b = pred_fn(X_valid[Im_num:Im_num+1,...])
		c = X_valid[Im_num:Im_num+1,...]
		#c[0,1,...] = c[0,0,...]
		b2 = pred_fn(c)
		b = b[0]
		b2 = b2[0]

		axarr[n,0].imshow(left_eye, cmap=cm.Greys_r)
		axarr[0,0].set_title('Left eye grayscale image')
		axarr[n,1].imshow(right_eye, cmap=cm.Greys_r)
		axarr[0,1].set_title('Right eye grayscale image')
		axarr[n,2].imshow(y_valid[Im_num,0,...])
		axarr[0,2].set_title('DepthMap ground truth')
		axarr[n,3].imshow(b[0,0,...])
		axarr[0,3].set_title('DepthMap prediction')

		n +=1
		#f.savefig('foo.png')

	f.savefig('examples.png')