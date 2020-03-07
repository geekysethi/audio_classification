import glob
import librosa
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torch


transform_train = transforms.Compose([transforms.ToTensor()])
transform_test = transforms.Compose([transforms.ToTensor()])


class read_dataset(Dataset):

	def __init__(self,file_path, labels_path, file_type, transforms=None):
		
		self.df = pd.read_csv(labels_path)
		self.file_path = file_path	
		self.file_type = file_type
		self.transforms = transforms


	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		
		if(self.file_type == "raw_file"):
			file_data = np.load(self.file_path+"/"+str(self.df["file_id"][index])+".npy")
			file_data = torch.from_numpy(file_data)
			file_data = file_data.type(torch.FloatTensor)
			file_data = file_data.view(file_data.size(0),1)
		
		else: 
			file_data = Image.open(self.file_path+"/"+str(self.df["file_id"][index])+".png")
			file_data = self.transforms(file_data)	
			
		
		label = self.df["label"][index]
		


		return (file_data, label)




def train_dataloader(file_path, labels_path, file_type, batch_size):

	train_set = read_dataset(file_path, labels_path, file_type, transform_train)

	return DataLoader(train_set, batch_size = batch_size, shuffle=True)


def test_dataloader(file_path, labels_path, file_type, batch_size):

	test_set = read_dataset(file_path, labels_path, file_type, transform_test)
	return DataLoader(test_set, batch_size=batch_size, shuffle=False)
