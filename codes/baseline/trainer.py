import wandb
import sys
sys.path.insert(1, 'models/')
from torchsummary import summary
import torch.optim as optim
import numpy as np
import torch 
import os
from utils import progress_bar,prepareData,normalized_confusion_matrix,plot_roc_curve,one_hot
import wandb
from torch.nn import functional as F
import torch.nn as nn
from models import select_model
import shutil
from PIL import Image


class Trainer():

	def __init__(self,config,train_loader,val_loader,model_name):
		
		self.config = config
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.model_name = model_name
		

		self.num_train = len(self.train_loader.dataset)
		self.num_valid = len(self.val_loader.dataset)
		self.saved_model_dir = self.config.saved_model_dir
		self.output_model_dir = self.config.output_model_dir
		self.best_model = config.best_model
		self.use_gpu = self.config.use_gpu



		if(self.config.resume == True):
			print("LOADING SAVED MODEL")
			self.net = select_model(self.model_name,self.config.file_type)
			self.loadCheckpoint()
		
		else:
			print("INTIALIZING NEW MODEL")
			
			self.net = select_model(self.model_name,self.config.file_type)
			
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.net = self.net.to(self.device)
		self.total_epochs = config.epochs
		

		if(self.model_name =="lstm_model" or self.model_name =="cnn_lstm_model"):
			print("NLL LOSS")
			self.criterion = nn.NLLLoss() 
			self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
			loss_name = "nll_loss"
			lr = 0.0001

		else:
			self.criterion = nn.CrossEntropyLoss()
			self.optimizer = optim.SGD(self.net.parameters(),lr = self.config.lr,
										momentum = self.config.momentum, 
										weight_decay = self.config.weight_decay)

			loss_name = "crossentropy_loss"
			lr = self.config.lr
			


		self.num_params = sum([p.data.nelement() for p in self.net.parameters()])
		self.batch_size = self.config.batch_size
		self.train_paitence = config.train_paitence
		self.num_classes = 10
		
		if(self.config.debug == False):
			self.experiment = wandb.init(project="audio_classification")
			
			hyper_params = {
							"model_name": self.model_name,
							"file_type": self.config.file_type,
							"dataset": self.config.dataset,
							"batch_size": self.config.batch_size,
   							"num_epochs": self.total_epochs,
							"loss_function": loss_name,
							"learning_rate": lr,
							"momentum":		self.config.momentum,
							"weight_decay": self.config.weight_decay

						}
			self.experiment.config.update(hyper_params)
			wandb.watch(self.net)

		# summary(self.net, input_size=(1, 128, 216))
		print('[*] Number of model parameters: {:,}'.format(self.num_params))
			
		
		

	def train(self):
		best_valid_acc=0

		print("\n[*] Train on {} sample pairs, validate on {} trials".format(
			self.num_train, self.num_valid))
		
		train_acc_list = []
		train_loss_list = []
		val_acc_list = []
		val_loss_list = []

		for epoch in range(0,self.total_epochs):

			print('\nEpoch: {}/{}'.format(epoch+1, self.total_epochs))
			

			train_acc,train_loss = self.train_one_epoch(epoch)
			val_acc,val_loss = self.val_one_epoch(epoch)

			train_acc_list.append(train_acc)
			train_loss_list.append(train_loss)
			val_acc_list.append(val_acc)
			val_loss_list.append(val_loss)
			# check for improvement
			if(val_acc > best_valid_acc):
				print("COUNT RESET !!!")
				best_valid_acc = val_acc
				self.counter = 0
				self.saveCheckPoint(
				{
					'epoch': epoch + 1,
					'model_state': self.net.state_dict(),
					'optim_state': self.optimizer.state_dict(),
					'best_valid_acc': best_valid_acc,
				},True)

			else:
				self.counter += 1
				
			
			if self.counter > self.train_paitence:
				self.saveCheckPoint(
				{
					'epoch': epoch + 1,
					'model_state': self.net.state_dict(),
					'optim_state': self.optimizer.state_dict(),
					'best_valid_acc': val_acc,
				},False)
				print("[!] No improvement in a while, stopping training...")
				print("BEST VALIDATION ACCURACY: ",best_valid_acc)


				break
		
		print("Saving Data ...")
		self.plot_cm_roc(self.val_loader)
		save_path = os.path.join(self.output_model_dir, self.config.model_name, self.config.file_type)
		if(self.config.debug == False):
			wandb.run.summary["best_accuracy"] = best_valid_acc
		np.save(os.path.join(save_path,"data/train_acc.npy"), np.array(train_acc_list))
		np.save(os.path.join(save_path,"data/train_loss.npy"),np.array(train_loss_list))
		np.save(os.path.join(save_path,"data/val_acc.npy"),np.array(val_acc_list))
		np.save(os.path.join(save_path,"data/val_loss.npy"),np.array(val_loss_list))
		
		# return train_acc_list,train_loss_list,val_acc_list,val_loss_list

		

	

	def train_one_epoch(self,epoch):
		self.net.train()
		train_loss = 0
		correct = 0
		total = 0

		for batch_idx, (file_data, targets) in enumerate(self.train_loader):

			file_data = file_data.to(self.device)
			targets = targets.to(self.device)
			
			self.optimizer.zero_grad()

			outputs = self.net(file_data)

			loss = self.criterion(outputs, targets)
			loss.backward()
			self.optimizer.step()

			train_loss += loss.item()
			currentTrainLoss=loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
			del(file_data)
			del(targets)
			
			progress_bar(batch_idx, len(self.train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

			if(self.config.debug == False):

				self.experiment.log({'Train/Loss': train_loss/(batch_idx+1)},step = epoch)
				self.experiment.log({"Train/Accuracy": correct / total},step =  epoch)

		return correct/total, currentTrainLoss

	

	def val_one_epoch(self,epoch):
		self.net.eval()
		validation_loss = 0
		correct = 0
		total = 0
		accuracy_list = []
		loss_list = []


		with torch.no_grad():
			for batch_idx, (file_data, targets) in enumerate(self.val_loader):
				
				
				file_data = file_data.to(self.device)
				targets = targets.to(self.device)
				outputs = self.net(file_data)
				loss = self.criterion(outputs, targets)
				validation_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()
				progress_bar(batch_idx, len(self.val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
				% (validation_loss/(batch_idx+1), 100.*correct/total, correct, total))

				del(file_data)
				del(targets)
			if(self.config.debug == False):
				
				self.experiment.log({'Validation/Loss': validation_loss/(batch_idx+1)},step =  epoch)
				self.experiment.log({"Validation/Accuracy": correct/total}, step = epoch)


				
		validation_accuracy = correct/total
		return validation_accuracy, validation_loss/(batch_idx+1)



	def test(self,dataLoader):
		self.net.eval()
		test_loss = 0
		correct = 0
		total = 0

		probOutput=[]
		trueOutput=[]	
		predictedOutput = []
		

		with torch.no_grad():
			for batch_idx, (file_data, targets) in enumerate(dataLoader):

				
				file_data = file_data.to(self.device)
				targets = targets.to(self.device)
				outputs = self.net(file_data)

				_, predicted = outputs.max(1)
				tempOutput = F.softmax(outputs,dim=1).cpu().numpy()
				probOutput.append(tempOutput)

				trueOutput.append(targets.cpu().numpy())
				predictedOutput.append(predicted.cpu().numpy())

				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()
				
				
				del(file_data)
				del(targets)

		print("\nTEST ACCURACY: ",100.*correct/total)	
		if(self.config.debug == False):
			wandb.run.summary["test_accuracy"] = correct/total

		return probOutput,trueOutput,predictedOutput
		

		
	def saveCheckPoint(self,state,isBest):
		filename = "model.pth"
		ckpt_path = os.path.join(self.output_model_dir,self.config.model_name,self.config.file_type, filename)
		torch.save(state, ckpt_path)
		
		if isBest:

			filename = "best_model.pth"
			shutil.copyfile(ckpt_path, os.path.join(self.output_model_dir,self.config.model_name,self.config.file_type, filename))


	def loadCheckpoint(self):

		print("[*] Loading model from {}".format(self.saved_model_dir))
		if(self.best_model):
			print("LOADING BEST MODEL")

			filename = "best_model.pth"

		else:
			filename = "model.pth"

		ckpt_path = os.path.join(self.saved_model_dir,self.model_name,self.config.file_type, filename)
		print(ckpt_path)
		
		if(self.use_gpu==False):
			self.net=torch.load(ckpt_path, map_location=lambda storage, loc: storage)

		else:
			print("*"*40+" LOADING MODEL FROM GPU "+"*"*40)
			self.ckpt = torch.load(ckpt_path)
			self.net.load_state_dict(self.ckpt['model_state'])
			self.net.cuda()


	def plot_cm_roc(self,data_loader):
		print("*****************Calculating Confusion Matrix*****************")
		
		
		save_path = os.path.join(self.output_model_dir, self.config.model_name, self.config.file_type,"data/")
		
		classes = np.arange(self.num_classes)
		probOutput,trueOutput,predictedOutput = self.test(data_loader)
		

		trueOutput = prepareData(trueOutput)
		predictedOutput = prepareData(predictedOutput)
		probOutput = prepareData(probOutput)

		one_hot_true_out = one_hot(trueOutput)

		normalized_confusion_matrix(trueOutput,predictedOutput,classes,save_path)
		plot_roc_curve(one_hot_true_out,probOutput,classes,save_path)

		if(self.config.debug == False):
			path = os.path.join(self.output_model_dir, self.config.model_name, self.config.file_type,"data/")
		
			cm =  Image.open(path+"confusion_matrix.png")
			roc =  Image.open(path+"roc_curve.png")
			
			wandb.log({"Confusion Matrix": [wandb.Image(cm, caption="Confusion Matrix")]})
			wandb.log({"ROC Curve": [wandb.Image(roc, caption="ROC Curve")]})
			
			# wandb.sklearn.plot_confusion_matrix(trueOutput,predictedOutput,classes)
			# wandb.sklearn.plot_roc(one_hot_true_out,probOutput,classes)




		return None
