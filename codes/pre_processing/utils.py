import itertools
import random
import librosa
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io


def scale_minmax(X, min=0.0, max=1.0):
	X_std = (X - X.min()) / (X.max() - X.min())
	X_scaled = X_std * (max - min) + min
	return X_scaled

def spectrogram_image(y, sr, hop_length, n_mels,save_path):
	mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
	mels = np.log(mels + 1e-9) 
	
	img = scale_minmax(mels, 0, 255).astype(np.uint8)
	img = np.flip(img, axis=0) 
	img = 255-img # invert. make black==more energy

	# save as PNG
	skimage.io.imsave(save_path, img)

	
def mfcc_image(y, sr,save_path):
	mfcc_feat = librosa.feature.mfcc(y= y,sr = sr)
	d = librosa.amplitude_to_db(np.abs(mfcc_feat), ref=np.max)
 
	
	img = scale_minmax(d, 0, 255).astype(np.uint8)
	img = np.flip(img, axis=0) 
	img = 255-img 
	skimage.io.imsave(save_path, img)


def save_raw_vectors(y,save_path):
	
	np.save(save_path,y)





def load_audio_file(file_path):
	input_length = 110250
	data = librosa.core.load(file_path)[0] #, sr=16000
	if len(data)>input_length:
		data = data[:input_length]
	else:
		data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
	return data


def constant_length(data):
	input_length = 110250
	if len(data)>input_length:
		data = data[:input_length]
	else:
		data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
	return data



	
def plot_time_series(data,save_path):
	fig = plt.figure()
	plt.title('Raw wave ')
	plt.ylabel('Amplitude')
	plt.plot(np.linspace(0, 1, len(data)), data)
	plt.show()
	plt.savefig(save_path,dpi = 300)

# Noise
def add_noise(data):
	wn = np.random.randn(len(data))
	data_wn = data + 0.005*wn
	return data_wn

# Shifting the sound
def shift(data):

	data_roll = np.roll(data, 2000)
	data_roll = constant_length(data_roll)
	return data_roll



def stretch(data, rate= 0.8):
	new_data = librosa.effects.time_stretch(data, rate)
	new_data = constant_length(new_data)
	return new_data

def augment_data(data):
	new_data = []
	new_data.append(scale_minmax(data))
	new_data.append(scale_minmax(add_noise(data)))
	new_data.append(scale_minmax(shift(data)))
	new_data.append(scale_minmax(stretch(data)))
	
	return new_data


