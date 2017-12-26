from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
	"""Wrapper class for model hyperparameters."""

	def __init__(self):
		"""Sets the default model hyperparameters."""
		# Convolutional Layer 1.
		self.filter_size1 = 3 
		self.num_filters1 = 32

		# Convolutional Layer 2.
		self.filter_size2 = 3
		self.num_filters2 = 32

		# Convolutional Layer 3.
		self.filter_size3 = 3
		self.num_filters3 = 64

		# Number of color channels for the images: 1 channel for gray-scale.
		self.num_channels = 3

		# image dimensions (only squares for now)
		self.img_size = 240

		# Size of image when flattened to a single dimension
		self.img_size_flat = self.img_size * self.img_size * self.num_channels

		# Tuple with height and width of images used to reshape arrays.
		self.img_shape = (self.img_size, self.img_size)

		# batch size
		self.batch_size = 20

		# validation split
		self.validation_size = .2

		#num of parameteres
		self.num_param = 3

		# how long to wait after validation loss stops improving before terminating training
		self.early_stopping = None  # use None if you don't want to implement early stoping

		#Embedding Size 
		self.embedding_size = 20

		#LSTM Cell state size
		self.cell_state_size = 512

		#Vocab Size
		self.vocab_size = None


class TrainingConfig(object):
	"""Sets the default training hyperparameters."""


