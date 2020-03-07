import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from resnet_utils import BasicBlock, Bottleneck, _resnet
from densenet_utils import densenet121
import torch

def resnet18(pretrained=False, progress=True, **kwargs):

	return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
				   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
	
	return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
				   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
	
	return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
				   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):

	return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
				   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
	return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
				   **kwargs)

def resnet_feature_model():

	model = resnet18()
	model = nn.Sequential(*list(model.classifier.children())[:-3])
	return model

class LSTM_2(nn.Module):
	def __init__(self):
		super(LSTM_2, self).__init__()

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

	
	def forward(self, x):
		# input x : 23 x 59049 x 1
		# expected conv1d input : minibatch_size x num_channel x width

		x = x.view(x.shape[0], 1,-1)
		# print(x.size())
		# x : 23 x 1 x 59049

		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.conv5(out)
		out = self.conv6(out)
		out = self.conv7(out)
		out = self.conv8(out)
		out = self.conv9(out)
		out = self.conv10(out)
		out = self.conv11(out) 
		out,hidden = self.lstm(out)
		out = out.reshape(x.shape[0], out.size(1) * out.size(2))		
		out = self.fc(out)
		out = F.log_softmax(out, dim=1)


		#logit = self.activation(logit)

		return out




class LSTM(nn.Module):
	def __init__(self, input_dim,input_size, hidden_dim, batch_size, num_layers, output_dim):
		super(LSTM, self).__init__()
		self.input_dim = input_dim
		self.input_size = input_size
		self.hidden_dim = hidden_dim
		self.batch_size = batch_size
		self.num_layers = num_layers
		self.output_dim = output_dim
		
		

		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True, dropout = 0.4)
		self.linear = nn.Linear(self.hidden_dim, output_dim)

	def init_hidden(self):
		return (
			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
		)

	def forward(self, x):

		out,hidden = self.lstm(x)
		out = out[:,-1]
		# out = out.reshape(-1,out.size(1)*out.size(2))
		out = self.linear(out)
		out = F.log_softmax(out, dim=1)

		return out

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


		


	def forward(self, x):


		# hidden = self.init_hidden()
		features = self.cnn(x)
		features = features.view(-1,features.size(1),features.size(2)*features.size(3))
		features = features.permute(0, 2, 1)
		out, hidden = self.lstm(features)
		out = out.reshape(-1,out.size(1)*out.size(2))
		out = self.linear(out)
		out = F.log_softmax(out, dim=1)
		return out
		
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
			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
			torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
		)



	def get_block_size(self, layer):
		return layer[-1][-1].bn2.weight.size()[0]



class Ensemble_model(nn.Module):
	def __init__(self):
		super(Ensemble_model, self).__init__()
		self.spec_net = resnet18()
		self.mfcc_net = resnet18()
	


	def forward(self, spec_file, mfcc_file):
		spec_out = self.spec_net(spec_file)
		mfcc_out = self.mfcc_net(mfcc_file)
		
		mean_output = torch.mean(torch.stack((spec_out,mfcc_out)),dim = 0)

		return spec_out


def select_model(model_name,data_type):

	if(model_name == "cnn"):
		
		print("RESNET !!!")
		# model = resnet18()
		model = densenet121()
	
	
	elif(model_name == "lstm"):

		print("Build LSTM RNN model ...")
		input_size = 1
		hidden_size = 128
		output_size = 10
		batch_size = 5
		n_layers = 2
		seq_len = 110250

		model = LSTM_2()


	elif("cnn_lstm"):

		print("Build CNN LSTM model ...")

		if(data_type == "spec"):
			print("FOR SPECTROGRAM DATA ...")
			temp = 28
		else:
			print("FOR MFCC DATA ...")

			temp = 7

		hidden_dim = 128
		batch_size = 2
		num_layers = 2
		output_dim = 10
	
		model = CRNN(hidden_dim,batch_size,num_layers,output_dim,temp)

	return model





# model = select_model("lstm_model")
# x = torch.randn((2,110250,1))

# model(x)