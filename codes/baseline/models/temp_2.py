import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):
	def __init__(self, input_dim,input_size, hidden_dim, batch_size, num_layers, output_dim):
		super(LSTM, self).__init__()
		self.input_dim = input_dim
		self.input_size = input_size
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.output_dim = output_dim
		
		

		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True)
		self.linear = nn.Linear(self.hidden_dim, output_dim)

	def init_hidden(self):
		return (
			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
		)

	def forward(self, x):

		out,hidden = self.lstm(x)
		print(out.size())
		out = out[:,-1]
		print(out.size())
		# out = out.reshape(-1,out.size(1)*out.size(2))
		
		out = self.linear(out)
		out = F.log_softmax(out, dim=1)

		return out

input_dim = 1
input_size = 2560
hidden_dim = 128
batch_size =2
num_layers = 2
output_dim = 10


print("Build LSTM RNN model ...")
model = LSTM(input_dim,input_size, hidden_dim, batch_size, num_layers, output_dim)	

# model = select_model("lstm_model")
x = torch.randn((batch_size,input_size,input_dim))
print(x.size())

model(x)