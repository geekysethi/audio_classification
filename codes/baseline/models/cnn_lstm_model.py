import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, n_layers = 2):
		super(LSTM, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.conv1 = nn.Conv1d(input_size, hidden_size, 4)
		self.pool = nn.AvgPool1d(3)

		self.cnn = nn.Sequential(nn.Conv1d(hidden_size, hidden_size, 1),
								nn.ReLU(),
								nn.AvgPool1d(3))
		
		self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=0.01,batch_first = True)
		self.fc = nn.Linear(21776, output_size)

	def forward(self, x):
		
		x = x.permute(0,2,1)
		print(x.size())
		out = F.relu(self.conv1(x))
		print(out.size())
		out = self.pool(out)
		print(out.size())

		out = self.cnn(out)
		print(out.size())

		out = self.cnn(out)
		print(out.size())
		
		out = self.cnn(out)
		print(out.size())

		out = out.permute(0,2,1)
		print(out.size())



		out = torch.tanh(out)
		out, hidden = self.lstm(out)

		# print(out.size())

		out = out.reshape(-1,out.size(1)*out.size(2))

		print(out.size())

		out = self.fc(out)
		print(out.size())
		out = F.log_softmax(out, dim=1)
		return out

		

input_size = 1
hidden_size = 16
output_size = 10
batch_size = 5
n_layers = 2
seq_len = 110250

model = LSTM(input_size, hidden_size, output_size, n_layers=n_layers)

inputs = Variable(torch.rand(batch_size, seq_len,input_size)) # seq_len x batch_size x 
print(inputs.size())
model(inputs)
# print('outputs', outputs.size()) # conv_seq_len x batch_size x output_size
# print('hidden', hidden.size()) # n_layers x batch_size x hidden_size