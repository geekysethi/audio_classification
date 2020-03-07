import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class SampleCNN(nn.Module):
	def __init__(self):
		super(SampleCNN, self).__init__()

		# 59049 x 1
		self.conv1 = nn.Sequential(
									nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
									nn.BatchNorm1d(128),
									nn.ReLU())
		# 19683 x 128
		self.conv2 = nn.Sequential(
									nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm1d(128),
									nn.ReLU(),
									nn.MaxPool1d(3, stride=3))
		# 6561 x 128
		self.conv3 = nn.Sequential(
									nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm1d(128),
									nn.ReLU(),
									nn.MaxPool1d(3,stride=3))
		# 2187 x 128
		self.conv4 = nn.Sequential(
									nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm1d(256),
									nn.ReLU(),
									nn.MaxPool1d(3,stride=3))
		# 729 x 256
		self.conv5 = nn.Sequential(
									nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm1d(256),
									nn.ReLU(),
									nn.MaxPool1d(3,stride=3))
		# 243 x 256
		self.conv6 = nn.Sequential(
									nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm1d(256),
									nn.ReLU(),
									nn.MaxPool1d(3,stride=3),
									nn.Dropout(0.4))
		# 81 x 256
		self.conv7 = nn.Sequential(
									nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm1d(256),
									nn.ReLU(),
									nn.MaxPool1d(3,stride=3))
		# 27 x 256
		self.conv8 = nn.Sequential(
									nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm1d(256),
									nn.ReLU(),
									nn.MaxPool1d(3,stride=3))

		# 9 x 256
		self.conv9 = nn.Sequential(
									nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm1d(256),
									nn.ReLU(),
									nn.MaxPool1d(3,stride=3))
		
		# 3 x 256
		self.conv10 = nn.Sequential(
									nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
									nn.BatchNorm1d(512),
									nn.ReLU(),
									nn.MaxPool1d(3,stride=3))
		# 1 x 512 
		self.conv11 = nn.Sequential(
								nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
								nn.BatchNorm1d(512),
								nn.ReLU(),
								nn.Dropout(0.4))
		# 1 x 512 
		self.lstm = nn.LSTM(1, 128, 2, dropout=0.01,batch_first = True)
		self.fc = nn.Linear(512*128, 10)

		# self.activation = nn.Sigmoid()
	
	def forward(self, x):
		# input x : 23 x 59049 x 1
		# expected conv1d input : minibatch_size x num_channel x width

		x = x.view(x.shape[0], 1,-1)
		print(x.size())
		# x : 23 x 1 x 59049

		out = self.conv1(x)
		print(out.size())
		out = self.conv2(out)
		print(out.size())
		out = self.conv3(out)
		print(out.size())
		out = self.conv4(out)
		print(out.size())
		out = self.conv5(out)
		print(out.size())
		out = self.conv6(out)
		print(out.size())
		out = self.conv7(out)
		print(out.size())
		out = self.conv8(out)
		print(out.size())
		out = self.conv9(out)
		print(out.size())

		out = self.conv10(out)
		print(out.size())

		out = self.conv11(out) 
		print(out.size())

		out,hidden = self.lstm(out)
		print(out.size())
		
		out = out.reshape(x.shape[0], out.size(1) * out.size(2))
		print(out.size())
		
		out = self.fc(out)
		print(out.size())

		#logit = self.activation(logit)

		return out


input_size = 1
hidden_size = 16
output_size = 10
batch_size = 5
n_layers = 2
seq_len = 110250

# model = LSTM(input_size, hidden_size, output_size, n_layers=n_layers)
model = SampleCNN()
inputs = Variable(torch.rand(batch_size, seq_len,input_size)) # seq_len x batch_size x 
print(inputs.size())
model(inputs)