import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import lasagne.layers

import nn_model


if __name__ == "__main__":

	X_train,X_left_train,X_right_train, y_train, X_val,X_left_val,X_right_val, y_val = main(model='cnn', num_epochs=45)

	X_train,X_left_train,X_right_train, y_train, X_valid,X_left_valid,X_right_valid, y_valid = nn_model.load_StereoImages()
	print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)


	plt.imshow(y_valid[11,0,...], cmap=cm.Greys_r)
	plt.show()
	plt.imshow(X_valid[11,0,...], cmap=cm.Greys_r)
	plt.show()


	input_var = T.tensor4('inputs', dtype=theano.config.floatX)
	network = nn_model.build_stereo_cnn(input_var)
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