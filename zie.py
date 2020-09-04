from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
# import tensorflow as tf
import collections
from scipy import sparse
import sys
from graph_builder import GraphBuilder
import warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


def separate_valid(reviews, frac):
	# separate frac of the trainset to validate and check for convergence
	review_size = reviews['scores'].shape[0]
	vind = np.random.choice(review_size, int(frac * review_size), replace=False)
	tind = np.delete(np.arange(review_size), vind)

	trainset = dict(scores=reviews['scores'][tind, :], atts=reviews['atts'][tind])
	validset = dict(scores=reviews['scores'][vind, :], atts=reviews['atts'][vind])
	
	return trainset, validset


def validate(valid_reviews, session, inputs, outputs):
	# compute validation log-likelihood and report mean across the set
	valid_size = valid_reviews['scores'].shape[0]
	ins_llh = np.zeros(valid_size)
	for iv in range(valid_size): 
		atts, indices, labels = generate_batch(valid_reviews, iv)  # random row from validation set
		if indices.size <= 1:
			raise Exception('in validation set: row %d has only less than 2 non-zero entries' % iv)
		feed_dict = {inputs['input_att']: atts, inputs['input_ind']: indices, inputs['input_label']: labels}
		ins_llh[iv] = session.run((outputs['llh']), feed_dict=feed_dict)
	
	mv_llh = np.mean(ins_llh)
	return mv_llh


def fit_emb(reviews, config, prt=True):
	np.random.seed(27)

	# this options is only related to training speed. 
	config.update(dict(sample_ratio=0.1))

	# the model takes out 1/10 of the training data to show validation log-likelihood
	# used to check for convergence of optimization
	use_valid_set = True 
	if use_valid_set:
		reviews, valid_reviews = separate_valid(reviews, 0.1)

	graph = tf.Graph()
	with graph.as_default(): # returns a context manager that makes this specific Graph the default graph
							 # graph to which all operations will be added if you don’t explicitly create a new graph.
		tf.set_random_seed(27)
		
		builder = GraphBuilder()
		inputs, outputs, model_param = builder.construct_model_graph(reviews, config, init_model=None, training=True)

		''' Adagrad: It adapts the learning rate to the parameters, performing smaller updates 
			(i.e. low learning rates) for parameters associated with frequently occurring features,
			and larger updates (i.e. high learning rates) for parameters associated with infrequent features. 
			For this reason, it is well-suited for dealing with sparse data. 
			From https://ruder.io/optimizing-gradient-descent/index.html#adagrad'''
		# gradient step (compute and apply gradients)
		optimizer = tf.train.AdagradOptimizer(0.05).minimize(outputs['objective']) # paper says learning_rate is 0.1 (?)
		init = tf.global_variables_initializer()

	with tf.Session(graph=graph) as session:

		# Visualize the DAG with tensorboard --logdir="output"
		writer = tf.summary.FileWriter("output", session.graph)
		writer.close()

		# We must initialize all variables before we use them.
		init.run()

		nprint = 5000 # print loss every nprint steps
		val_accum = np.array([0.0, 0.0]) # validation sums of training llh and objective
		train_logg = np.zeros([int(config['max_iter'] / nprint) + 1, 3]) # logg of llh, obj, and debug llh at every nprint steps

		review_size = reviews['scores'].shape[0] # the subset of training data not used for validations
												 # contains both scores and features (atts, covariates)

		# main training loop, max_iter steps done
		for step in range(1, config['max_iter'] + 1):

			rind = np.random.choice(review_size)
			atts, indices, labels = generate_batch(reviews, rind) # random row from training set
			if indices.size <= 1: # neglect rows with only one entry
				raise Exception('Row %d of the data has only one non-zero entry.' % rind)
			feed_dict = {inputs['input_att']: atts, inputs['input_ind']: indices, inputs['input_label']: labels}

			_, llh_val, obj_val, debug_val = session.run((optimizer, outputs['llh'], outputs['objective'], outputs['debugv']),
												feed_dict=feed_dict) # returns same shape as what is inputed
			# training log-likelihood, model objective (neg llh + regularizer), validation log-likelihood
			val_accum = val_accum + np.array([llh_val, obj_val])

			# print loss every nprint iterations
			if step % nprint == 0 or np.isnan(llh_val) or np.isinf(llh_val):
				
				valid_llh = 0.0
				break_flag = False
				if use_valid_set:
					valid_llh = validate(valid_reviews, session, inputs, outputs)
					#if ivalid > 0 and valid_llh[ivalid] < valid_llh[ivalid - 1]: # check if performance becomes worse
					#    print('validation llh: ', valid_llh[ivalid - 1], ' vs ', valid_llh[ivalid])
					#    break_flag = True
				
				# record the three values 
				ibatch = int(step / nprint) # 0 for steps 1-nprint, 1 for nprint+1-2nprint ...
				train_logg[ibatch, :] = np.append(val_accum / nprint, valid_llh)
				val_accum[:] = 0.0 # reset the accumulator
				print("iteration[", step, "]: average llh, obj, and valid_llh are ", train_logg[ibatch, :])
				
				# check convergence with debug (validation) log-likelihood
				if np.isnan(llh_val) or np.isinf(llh_val):
					print('Loss value is ', llh_val, ', and the debug value is ', debug_val)
					raise Exception('Bad values')
   
				if break_flag:
					break

		# training done, save model parameters to dict for evaluation
		model = dict(alpha=model_param['alpha'].eval(), 
					   rho=model_param['rho'].eval(), 
					 invmu=model_param['invmu'].eval(), 
					weight=model_param['weight'].eval(), 
					   nbr=model_param['nbr'].eval())

		return model, train_logg


def evaluate_emb(reviews, model, config):

	graph = tf.Graph()
	with graph.as_default():
		tf.set_random_seed(27)
		# construct model graph
		print('Building graph...')
		builder = GraphBuilder()
		inputs, outputs, model_param = builder.construct_model_graph(reviews, config, model, training=False)
		init = tf.global_variables_initializer()

	with tf.Session(graph=graph) as session:
		# We must initialize all variables before we use them.
		print('Initializing...')
		init.run()

		llh_array = [] # llh over all entries
		pos_llh_array = [] # llh over only positive entries
		review_size = reviews['scores'].shape[0]
		print('Calculating llh of instances...')
		for step in range(review_size):
			att, index, label = generate_batch(reviews, step)
			if index.size <= 1: # neglect views with only one entry
				continue
			feed_dict = {inputs['input_att']: att, inputs['input_ind']: index, inputs['input_label']: label}
			ins_llh_val, pos_llh_val = session.run((outputs['ins_llh'], outputs['pos_llh']), feed_dict=feed_dict)

			# if step == 0:
			#    predicts = session.run(outputs['debugv'], feed_dict=feed_dict)
			#    print('%d movies ' % predicts.shape[0])
			#    print(predicts)

			llh_array.append(ins_llh_val)
			pos_llh_array.append(pos_llh_val)

		
		llh_array = np.concatenate(llh_array, axis=0)
		pos_llh_array = np.concatenate(pos_llh_array, axis=0)

		
		return llh_array, pos_llh_array


def generate_batch(reviews, rind, prt=True):
	# get random row from reviews
	atts = reviews['atts'][rind, :]
	_, ind, rate = sparse.find(reviews['scores'][rind, :]) # rows, cols, and values of nonzero entries
	# ind corresponds to column index (within the score row)
	# atts is not sparse so no need to use sparse.find
	if prt and rind<2: print("Generated batch:", rind, atts, ind, rate)
	return atts, ind, rate 
