from torchsummary import summary
import torch
import torch.nn as nn


class LSTM_model(nn.Module):

	def __init__(self,input_dim,hidden_dim,n_layers,output_dim):
		super(LSTM_model,self).__init__()
		
		self.n_layers = n_layers
		self.hidden_dim = hidden_dim

		self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.n_layers, batch_first=True)

		
		# self.fc =  nn.Linear(hidden_dim, output_dim)
	
	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
				weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
		return hidden
	
	def forward(self,x,hidden):
		# x = x.long()
		out = self.lstm(x)
		lstm_out, hidden = self.lstm(out, hidden)
		# lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
		
		# out = out[0]
		# print(out.size())
	
		# out = out.view(64,out.size(1)*out.size(2))
		# print(out.size())
		# print(out[1].size())
		

		# out = self.fc(out)
		# out = torch.nn.functional.log_softmax(out)

		return out,hidden




input_dim = 1
hidden_dim = 5
n_layers = 2
output_dim = 10
batch_size = 64
model = LSTM_model(input_dim,hidden_dim,n_layers,output_dim)

h = model.init_hidden(batch_size)
h = tuple([e.data for e in h])
# # model.cuda()
x = torch.randn((64,100,input_dim))
# # print(x.size())
model(x,h)
# model(spec_file,mfcc_file)
# print(mfcc_file.size())
# summary(model,(1,input_dim))