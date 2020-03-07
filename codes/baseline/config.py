import os
from pathlib import Path


class config_class():
	def __init__(self,resume,debug,file_type,model_name):

		


		self.file_type = file_type
		self.model_name = model_name

		self.dataset = "base" #"urbansound", "augmentd", "base"
		# self.saved_model_dir = "./outputs/urbansound_models/"
		# self.output_model_dir = "./outputs/"+self.dataset+"_fintune_models/"

		self.saved_model_dir = "./outputs/base_models/"
		self.output_model_dir = "./outputs/base_models/"
		

		print("DATASET SELECTED: ",self.dataset.upper())



		if(self.file_type == "raw_file"):
			if(self.dataset == "augmented"):
		
				self.train_file_path = "../../data/augmented_data/train/raw_vectors"
				self.val_file_path = "../../data/augmented_data/val/raw_vectors/"


			elif(self.dataset == "urbansound"):

				self.train_file_path = "../../data/UrbanSound8K/raw_vectors"
				self.val_file_path = "../../data/UrbanSound8K/raw_vectors/"


			elif(self.dataset == "base"):

				self.train_file_path = "../../data/training_data/train/raw_vectors"
				self.val_file_path = "../../data/training_data/val/raw_vectors/"




		elif(self.file_type == "mfcc"):

			if(self.dataset == "augmented"):
				self.train_file_path = "../../data/augmented_data/train/mfcc"
				self.val_file_path = "../../data/augmented_data/val/mfcc/"


			elif(self.dataset == "urbansound"):

				self.train_file_path = "../../data/UrbanSound8K/mfcc"
				self.val_file_path = "../../data/UrbanSound8K/mfcc/"

			
			elif(self.dataset == "base"):

				self.train_file_path = "../../data/training_data/train/mfcc"
				self.val_file_path = "../../data/training_data/val/mfcc/"



		elif(self.file_type == "spec"):

			if(self.dataset == "augmented"):
				self.train_file_path = "../../data/augmented_data/train/spec"
				self.val_file_path = "../../data/augmented_data/val/spec/"


			elif(self.dataset == "urbansound"):

				self.train_file_path = "../../data/UrbanSound8K/spec"
				self.val_file_path = "../../data/UrbanSound8K/spec/"



			
			elif(self.dataset == "base"):
				self.train_file_path = "../../data/training_data/train/spectrogram"
				self.val_file_path = "../../data/training_data/val/spectrogram"


		if(self.dataset == "augmented"):
			self.train_labels_path = "../../data/augmented_data/train/train.csv"
			self.val_labels_path = "../../data/augmented_data/val/val.csv"


		elif(self.dataset == "urbansound"):
			self.train_labels_path = "../../data/UrbanSound8K/train.csv"
			self.val_labels_path = "../../data/UrbanSound8K/val.csv"


		
		elif(self.dataset == "base"):

				self.train_labels_path = "../../data/training_data/train/train.csv"
				self.val_labels_path = "../../data/training_data/val/val.csv"


		self.epochs = 200
		self.batch_size = 16
		self.numWorkers = 2
		self.lr=0.001
		self.momentum=0.9
		self.weight_decay=5e-4
		self.train_paitence=20



		self.resume=resume
		self.debug = debug
		self.use_pretrained=True
		self.best_model = True
		self.use_gpu=True