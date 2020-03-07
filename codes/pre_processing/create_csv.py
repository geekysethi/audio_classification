import numpy as np
import pandas as pd
import shutil
import glob

total_per_class = 30
total_classes = 10

train = []
validation = []

def main():
	for current_class in range(total_classes):
		print("*"*80)
		print(current_class)

		for i in range(1,total_per_class+1):
			# print("+"*40)
			current_wav_path = "../../data/raw_data/wav_files/"+str(i)+"_"+str(current_class)+".wav"
			current_mfcc_path = "../../data/raw_data/mfcc/"+str(i)+"_"+str(current_class)+".png"
			current_spectrogram_path = "../../data/raw_data/spectrogram/"+str(i)+"_"+str(current_class)+".png"
			current_file_id = str(i)+"_"+str(current_class)
			current_row = [current_file_id,current_class]
			print(current_file_id)

			if(i<=25):
				copy_wav_path =  "../../data/training_data/train/wav_files/"+str(i)+"_"+str(current_class)+".wav"
				copy_mfcc_path = "../../data/training_data/train/mfcc/"+str(i)+"_"+str(current_class)+".png"
				copy_spec_path = "../../data/training_data/train/spectrogram/"+str(i)+"_"+str(current_class)+".png"
				train.append(current_row)
				

			else:
				copy_wav_path =  "../../data/training_data/validation/wav_files/"+str(i)+"_"+str(current_class)+".wav"
				copy_mfcc_path = "../../data/training_data/validation/mfcc/"+str(i)+"_"+str(current_class)+".png"
				copy_spec_path = "../../data/training_data/validation/spectrogram/"+str(i)+"_"+str(current_class)+".png"
				validation.append(current_row)
			
			shutil.copy(current_wav_path,copy_wav_path)
			shutil.copy(current_mfcc_path,copy_mfcc_path)
			shutil.copy(current_spectrogram_path,copy_spec_path)
			
			

		# break

	columns = ['file_id',"label"]
	print("[INFO] SAVING DATA IN DATAFRAME")
	df = pd.DataFrame(index=np.arange(len(train)), columns=columns)
	df.loc[:] = train
	df.to_csv("../../data/training_data/train/train.csv", encoding="utf-8", index=False)



	df = pd.DataFrame(index=np.arange(len(validation)), columns=columns)
	df.loc[:] = validation
	df.to_csv("../../data/training_data/validation/validation.csv", encoding="utf-8", index=False)

if __name__ == "__main__":
	main()