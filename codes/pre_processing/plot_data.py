import numpy as np
import librosa
import librosa.display
from matplotlib import pyplot as plt
import skimage
import glob 

def plot_data(feat,save_path):
	plt.figure()
	d = librosa.amplitude_to_db(np.abs(feat), ref=np.max)
	print(d.shape)
	print(mfcc_feat.shape)
	librosa.display.specshow(d, x_axis='time')
	ax = plt.subplot(111)
	ax.get_xaxis().set_visible(False)
	plt.tight_layout()
	plt.savefig(save_path,dpi = 300)
	plt.close()

def plot_wave_data(y,sr,save_path):
	plt.figure()
	plt.title("Audio waveform")
	librosa.display.waveplot(y, sr=sr)
	plt.savefig(save_path,dpi = 300)
	plt.close()



if __name__ == "__main__":
	

	file_paths = glob.glob("../../data/raw_data/wav_files/*")

	# print(len(file_paths))

	for current_file in file_paths:
		plt.figure()
		current_file_id = current_file.split("/")[5].split(".")[0]
		print(current_file_id)

		y,sr = librosa.load(current_file)
		mfcc_feat = librosa.feature.mfcc(y= y,sr = sr, n_mfcc=20)
		spec_feat = librosa.feature.melspectrogram(y=y, sr=sr)
		
		mfcc_save_path = "../../data/plots/mfcc/"+str(current_file_id)+".png"
		spec_save_path = "../../data/plots/spec/"+str(current_file_id)+".png"
		wave_save_path = "../../data/plots/waveforms/"+str(current_file_id)+".png"

		plot_data(mfcc_feat,mfcc_save_path)
		plot_data(spec_feat,spec_save_path)
		plot_wave_data(y,sr,wave_save_path)

		# break	


