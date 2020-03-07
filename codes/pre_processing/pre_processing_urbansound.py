import glob
import shutil
import librosa
import librosa.display
import numpy as np
import pandas as pd
import skimage
from matplotlib import cm
from matplotlib import pyplot as plt

from utils import mfcc_image, save_raw_vectors, spectrogram_image,load_audio_file

if __name__ == "__main__":
	hop_length = 512 
	n_mels = 128 
	time_steps = 384

	
	df = pd.read_csv("../../data/UrbanSound8K/metadata/UrbanSound8K.csv")

	for i in range(len(df)):
		current_row = df.iloc[i]
		
		current_file_path = "../../data/UrbanSound8K/audio/fold"+str(current_row["fold"])+"/"+str(current_row["slice_file_name"])
		file_id = current_row["slice_file_name"].split(".")[0]
		
		# print(file_id)
		mfcc_save_path = "../../data/UrbanSound8K/mfcc/"+str(file_id)+".png"
		spec_save_path = "../../data/UrbanSound8K/spec/"+str(file_id)+".png"
		raw_save_path = "../../data/UrbanSound8K/raw_vectors/"+str(file_id)+".npy"
		wav_save_path = "../../data/UrbanSound8K/wav_files/"+str(file_id)+".wav"
		
		
		print(current_file_path)
		y = load_audio_file(current_file_path)
		sr = 22050
		# print(y.shape)
		mfcc_image(y,sr,mfcc_save_path)
		spectrogram_image(y, sr=sr, hop_length=hop_length, n_mels=n_mels, save_path = spec_save_path)
		save_raw_vectors(y,raw_save_path)
		shutil.copyfile(current_file_path,wav_save_path)
