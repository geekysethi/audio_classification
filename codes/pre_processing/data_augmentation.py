from utils import augment_data,mfcc_image, spectrogram_image,save_raw_vectors,plot_time_series


import glob
import shutil
import librosa
import librosa.display
import numpy as np
import pandas as pd
import skimage
from matplotlib import cm
from matplotlib import pyplot as plt


if __name__ == "__main__":
	hop_length = 512 
	n_mels = 128 
	time_steps = 384

	flag = "val"
	file_list = glob.glob("../../data/training_data/"+str(flag)+"/wav_files/*")
	

	for current_file_path in file_list:
		
		file_id = current_file_path.split("/")[6].split(".")[0]

		mfcc_save_path = "../../data/augmented_data/"+str(flag)+"mfcc/"+str(file_id)+".png"
		spec_save_path = "../../data/augmented_data/"+str(flag)+"spec/"+str(file_id)+".png"
		raw_save_path = "../../data/augmented_data/"+str(flag)+"raw_vectors/"+str(file_id)+".npy"
		wav_save_path = "../../data/augmented_data/"+str(flag)+"wave_files/"+str(file_id)+".png"
				
		print(current_file_path)
		y, sr = librosa.load(current_file_path)
		augmented_data = augment_data(y)
		for i in range(4):

			mfcc_save_path = "../../data/augmented_data/"+str(flag)+"/mfcc/"+str(file_id)+"_"+str(i)+".png"
			spec_save_path = "../../data/augmented_data/"+str(flag)+"/spec/"+str(file_id)+"_"+str(i)+".png"
			raw_save_path = "../../data/augmented_data/"+str(flag)+"/raw_vectors/"+str(file_id)+"_"+str(i)+".npy"
			wav_save_path = "../../data/augmented_data/"+str(flag)+"/wav_files/"+str(file_id)+"_"+str(i)+".png"

			current_data = augmented_data[i]
			if(len(current_data) != 110250):
				print("*******************YOLO*********************")
			# temp_path = str(i)+".png"
			# plot_time_series(current_data,temp_path)

			librosa.output.write_wav(wav_save_path,current_data,sr)
			mfcc_image(current_data,sr,mfcc_save_path)
			spectrogram_image(current_data, sr=sr, hop_length=hop_length, n_mels=n_mels, save_path = spec_save_path)
			save_raw_vectors(current_data,raw_save_path)
			
		# break
			
