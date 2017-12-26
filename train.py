"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import cnn_lstm_model
import helpers
import time
import configuration
import matplotlib.pyplot as plt


TRAIN_DATA_DIR = '/media/nikhil/Data/Experiments/Handwritten text Extraction/handwriting Extraction Codes/ArabicTextRecognitionCodeTF/small_train'
DICT_DATA_DIR = '/media/nikhil/Data/Experiments/Handwritten text Extraction/handwriting Extraction Codes/ArabicTextRecognitionCodeTF/'

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
					   "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("train_data_dir", TRAIN_DATA_DIR,
					   "Directory for reading images for input.")
tf.flags.DEFINE_string("dict_data_dir", DICT_DATA_DIR,
					   "Directory for reading dictonary of Arabic words")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
						"Frequency at which loss and global step are logged.")

tf.logging.set_verbosity(tf.logging.INFO)


def main():
	model_config = configuration.ModelConfig()
	training_config = configuration.ModelConfig()

	model_config.vocab_size

	#Preparation of Arab Characters Set
	arab_char_set = helpers.prepare_arab_char_set(FLAGS.dict_data_dir)
	
	#Preparation of char2num and num2char mapping
	arab_char2num, arab_num2char = helpers.prepare_arab_char_dict(arab_char_set)

	#Load Images and its corresponding Words
	train_images, train_arabic_word, arabic_seq_max_length = helpers.load_train(FLAGS.train_data_dir, FLAGS.dict_data_dir, model_config.img_size)

	#print(train_images.get_shape())

	train_arabic_word_targets = [[arab_char2num[w_] for w_ in word]+ [arab_char2num['<GO>']] 
								for word in train_arabic_word]

	train_arabic_word_inputs = [[arab_char2num['<GO>']] + [arab_char2num[w_] for w_ in word]
								 for word in train_arabic_word]

	model_config.vocab_size = len(arab_char2num)


	input_seqs, target_seqs = helpers.next_feed(train_arabic_word_inputs, 
		train_arabic_word_targets, arab_char2num)

	
	#fd = {images : train_images, input_seqs: decoder_inputs_,
	#	targets_seqs: decoder_targets_}

	#Build the Tensorflow graph.
	g = tf.Graph()
	with g.as_default():
		#Build the Model.
		model = cnn_lstm_model.CNNLSTMModel(model_config, mode="train")
		model.build()


	
	with tf.Session(graph = g) as sess:
	    # initialize the session to generate the visualization file
	    sess.run(tf.global_variables_initializer())
	    
	    tvars = tf.trainable_variables()
	    tvars_vals = sess.run(tvars)
	    print(len(tvars))
	    
	    for var, val in zip(tvars, tvars_vals):
	        print(var.name)

	


	
	fd = {model.images : train_images, model.input_seqs: input_seqs,
		model.target_seqs: target_seqs}

	

	#fd = {model.input_seqs: input_seqs,
	#	model.target_seqs: target_seqs}



	sess = tf.InteractiveSession(graph=g)

	sess.run(tf.global_variables_initializer())

	print(sess.graph)
	print(tf.get_default_graph)

	loss_track = []

	epochs = 1000
	for epoch_i in range(epochs):
		start_time = time.time()
		#fd = next_feed()
		_, l = sess.run([model.train_op, model.loss], fd)
		loss_track.append(l)
		if epoch_i==0 or epoch_i % 100 ==0:
			print('Epoch {:3} Loss: {:>6.3f} Epoch duration: {:>6.3f}s'.format(epoch_i, l, 
																		  time.time() - start_time))
			predict_ = sess.run(model.decoder_prediction, fd)
			for i , (inp, pred) in enumerate(zip(fd[input_seqs.model].T[1:], predict_.T)):
				#inp_word = ''.join([arab_num2charY[x] for x in inp])
				#pred_word = ''.join([arab_num2charY[x] for x in pred])
				inp_word = [arab_num2char[x] for x in inp]
				pred_word = [arab_num2char[x] for x in pred]
				
				print('   sample {}'.format(i+1))
				print('     input    > {}'.format(inp_word))
				print('     predicted> {}'.format(pred_word))
				if i >=2:
					break
			print()

	plt.plot(loss_track)
	print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))



if __name__ == "__main__":
	tf.app.run()


