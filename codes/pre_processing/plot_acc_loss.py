import numpy as np 
from matplotlib import pyplot as plt

model_name = ["cnn"]

file_type = ["mfcc","spec"]
plot_values = {"mfcc":"Mfcc",
				"spec":"Sprectrogram"}

for current_model in model_name:
	plt.figure()
	for current_file_type in file_type:
		path = "../baseline/outputs/saved_models/"+str(current_model)+"/"+str(current_file_type)
		
		train_acc = np.load(path+"/data/train_acc.npy")
		train_loss = np.load(path+"/data/train_loss.npy")
		val_acc = np.load(path+"/data/val_acc.npy")
		val_loss = np.load(path+"/data/val_loss.npy")

		plt.plot(np.arange(len(train_acc)),train_acc)

		# print(path)
		# plt.figure()
		# t
		# plt.plot(np.arange())