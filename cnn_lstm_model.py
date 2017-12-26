from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

class CNNLSTMModel(object):
	"""Arabic Image to text
	"""

	def __init__(self, config, mode):

		"""Basic Setup.

		Args:
			config: Object containing configuration parameters.
			mode: "train", "eval" or "inference"
			train_inception: Whether the inception submodel variable
		"""

		assert mode in ["train", "eval", "inference"]
		self.config = config
		self.mode = mode

		#Reader for the input data
		#self.Reader = tf.TFRecordReader()

		# A float32 Tensor with shape [batch_size, height, width, channels].
		self.images = None

		# A int32 Tensor with shape [batch_size, ]
		self.input_seqs = None

		# A int32 Tensor with shape [batch_size, ]
		self.target_seqs = None

		# Global step Tensor.
		self.global_step = None


	def is_training(self):
		"""Returns true if the model is built for training mode."""
		return self.mode == "train"

		
	def build_inputs(self):
		"""Input prefetching, preprocessing and batching."

		Outputs:
		self.images
		self.input_seqs
		self.target_seqs
		"""
		if self.mode =="inference":
			#process image and insert batch dimensions.
			images = tf.placeholder(tf.float32, shape=[None, self.config.img_size, self.config.img_size, self.config.num_channels], name='images')
			input_seqs = None

			target_seqs = None

		else:
			
			images = tf.placeholder(tf.float32, shape=[None, self.config.img_size, self.config.img_size, self.config.num_channels], name='images')
			input_seqs = tf.placeholder(tf.int32, (None, None), 'input_seqs')
			target_seqs = tf.placeholder(tf.int32, (None, None), 'target_seqs')




		self.images = images
		self.input_seqs = input_seqs
		self.target_seqs = target_seqs


	def build_cnn(self):
		"""Builds the image CNN model subgraph and generates image embedding 
		to be fed to LSTM Decoder model.
		Function contains helper function to create CNN

		Inputs:
			self.images

		Outputs:
			self.layers: learned features
		"""

		def new_weights(shape):
			return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

		def new_biases(length):
			return tf.Variable(tf.constant(0.05, shape=[length]))


		def new_conv_layer(input,              # The previous layer.
						   num_input_channels, # Num. channels in prev. layer.
						   filter_size,        # Width and height of each filter.
						   num_filters,        # Number of filters.
						   use_pooling=True):  # Use 2x2 max-pooling.

			# Shape of the filter-weights for the convolution.
			# This format is determined by the TensorFlow API.
			shape = [filter_size, filter_size, num_input_channels, num_filters]

			# Create new weights aka. filters with the given shape.
			weights = new_weights(shape=shape)

			# Create new biases, one for each filter.
			biases = new_biases(length=num_filters)

			layer = tf.nn.conv2d(input=input,
								 filter=weights,
								 strides=[1, 1, 1, 1],
								 padding='SAME')

			layer += biases

			# Use pooling to down-sample the image resolution?
			if use_pooling:
				layer = tf.nn.max_pool(value=layer,
									   ksize=[1, 2, 2, 1],
									   strides=[1, 2, 2, 1],
									   padding='SAME')

			layer = tf.nn.relu(layer)

			return layer, weights

		def reshape(layer):
			num_features1 = layer.get_shape()[1:3].num_elements()
			num_features2 = layer.get_shape()[3:4].num_elements()
			print(num_features1)
			print(num_features2)
			layer_flat = tf.reshape(layer, [-1, num_features1, num_features2])

			return layer_flat


		with tf.variable_scope("cnn_learning"):
			#Layer 1
			layer_conv1, weights_conv1 = \
			new_conv_layer(input=self.images,
						   num_input_channels=self.config.num_channels,
						   filter_size=self.config.filter_size1,
						   num_filters=self.config.num_filters1,
						   use_pooling=True)
			
			#Layer 2
			layer_conv2, weights_conv2 = \
			new_conv_layer(input=layer_conv1,
						   num_input_channels=self.config.num_filters1,
						   filter_size=self.config.filter_size2,
						   num_filters=self.config.num_filters2,
						   use_pooling=True)
			
			#Layer 3
			layer_conv3, weights_conv3 = \
			new_conv_layer(input=layer_conv2,
						   num_input_channels=self.config.num_filters2,
						   filter_size=self.config.filter_size3,
						   num_filters=self.config.num_filters3,
						   use_pooling=True)

			layer_flat = reshape(layer_conv3)


		self.layer_flat = layer_flat


	def build_seq_embeddings(self):
		"""Builds the input sequence embeddings.

		Inputs:
		self.input_seqs, self.target_seqs

		Outputs:
		self.seq_embeddings
		"""

		with tf.variable_scope("input_embedding"):
			input_embedding_map = tf.Variable(tf.random_uniform(name = 'input_embedding_map', 
				shape = [self.config.vocab_size, self.config.embedding_size]))
			input_embeddings = tf.nn.embedding_lookup(input_embedding_map, self.input_seqs)

		with tf.variable_scope("target_embedding"):
			target_embedding_map = tf.Variable(tf.random_uniform(name = 'target_embedding_map', 
				shape = [self.config.vocab_size, self.config.embedding_size]))
			target_embeddings = tf.nn.embedding_lookup(input_embedding_map, self.target_seqs)


		self.input_embeddings = input_embeddings
		self.target_embeddings = target_embeddings


	def build_model(self):
		"""Take input from cnn model and outputs loss.

		Inputs:
			self.seq_embeddings,
			self.target_seqs

		Outputs:
			self.total_loss (training and eval only)
			self.target_cross_entropy_losses(training and eval only)
			self.target_cross_entropy_losses_weights(training and eval only)
		"""

		def encoder(seqs, w_reuse=None):
			with tf.variable_scope("encoder", reuse=w_reuse):
				_, state = tf.nn.dynamic_rnn(
								tf.contrib.rnn.LSTMCell(self.config.cell_state_size),
								seqs, 
								dtype = tf.float32
							)
				 
			return state

		def decoder(output_embed, state, w_reuse=None):
			with tf.variable_scope("decoder", reuse=w_reuse):
				outputs, _ = tf.nn.dynamic_rnn(
								tf.contrib.rnn.LSTMCell(self.config.cell_state_size),
								output_embed, 
								dtype = tf.float32,
								initial_state=state,
								#time_major = False
							)
			return outputs


		with tf.variable_scope("encoder_decoder"):
			#Feed last layer of cnn_learning to encoder part
			last_state = encoder(self.layer_flat)

			#Feed input_seqs embedding to decoder part 
			decoder_outputs = decoder(self.input_embeddings, last_state)

		self.decoder_outputs = decoder_outputs

		with tf.variable_scope("logits"):
			logits = tf.contrib.layers.fully_connected(
				self.decoder_outputs, 
				num_outputs=self.config.vocab_size, 
				activation_fn=None)

		self.decoder_prediction = tf.argmax(logits, 2)


		#if self.mode == 'inference':


		#Compute losses
		with tf.variable_scope("losses"):
			stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
				labels = tf.one_hot(self.target_seqs, depth = self.config.vocab_size, dtype = tf.float32),
				logits = logits)

			loss = tf.reduce_mean(stepwise_cross_entropy)
			train_op = tf.train.AdamOptimizer().minimize(loss)

		self.loss = loss
		self.train_op = train_op


	def setup_global_step(self):
		"""Sets up the global step Tensor."""
		global_step = tf.Variable(
			initial_value=0,
			name="global_step",
			trainable=False,
			collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

		self.global_step = global_step


	def build(self):
		"""Creates all ops for training and evaluation."""
		self.build_inputs()
		self.build_cnn()
		self.build_seq_embeddings()
		self.build_model()
		self.setup_global_step()

