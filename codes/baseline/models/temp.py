import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.models as models
import string
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from resnet_utils import BasicBlock, Bottleneck, _resnet
import torch

def resnet18(pretrained=False, progress=True, **kwargs):

	return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
				   **kwargs)


# class LSTM(nn.Module):
# 	def __init__(self, input_dim,input_size, hidden_dim, batch_size, num_layers, output_dim):
# 		super(LSTM, self).__init__()
# 		self.input_dim = input_dim
# 		self.input_size = input_size
# 		self.hidden_dim = hidden_dim
# 		self.batch_size = batch_size
# 		self.num_layers = num_layers
# 		self.output_dim = output_dim

		
		

# 		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
# 		self.linear = nn.Linear(self.hidden_dim*self.input_size, output_dim)

# 	def init_hidden(self):
# 		return (
# 			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
# 			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
# 		)

# 	def forward(self, x):

# 		out, hidden = self.lstm(x)
# 		out = out.view(-1,out.size(1)*out.size(2))
# 		out = self.linear(out)
# 		out = F.log_softmax(out, dim=1)
# 		return out







class CRNN(nn.Module):
	def __init__(self,hidden_dim ,
					batch_size ,
					num_layers,
					output_dim,
					temp,
					dropout=0):

		super(CRNN, self).__init__()

		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.output_dim = output_dim
		self.dropout = dropout


		self.cnn = self.feature_extractor()


		self.lstm = nn.LSTM(self.get_block_size(self.cnn),self.hidden_dim, self.num_layers,batch_first=True,dropout=self.dropout)
		
		self.linear = nn.Linear(self.hidden_dim*temp, output_dim)


		


	def forward(self, x, decode=False):


		hidden = self.init_hidden()
		print(hidden[0].size())
		features = self.cnn(x)

		print(features.size())

		features = features.view(-1,features.size(1),features.size(2)*features.size(3))
	
		print(features.size())
		features = features.permute(0, 2, 1)
		print(features.size())


		out, hidden = self.lstm(features, hidden)
		print(out.size())
		out = out.reshape(-1,out.size(1)*out.size(2))
		out = self.linear(out)
		print(out.size())

	def feature_extractor(self):
		
		model = resnet18()
		new_model = nn.Sequential(model.conv1,
									model.bn1,
									model.relu,
									model.maxpool,
									model.layer1,
									model.layer2,
									model.layer3,
									model.layer4			
								)

		return new_model



	def init_hidden(self):
		return (
			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
		)



	def get_block_size(self, layer):
		return layer[-1][-1].bn2.weight.size()[0]




hidden_dim = 128
batch_size = 64
num_layers = 2
output_dim = 10
temp = 7

model = CRNN(hidden_dim,batch_size,num_layers,output_dim,temp)
x = torch.randn((64,1,20,216))
print(x.size())
model(x)

