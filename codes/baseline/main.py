from trainer import Trainer
from data_loader import train_dataloader, test_dataloader
from utils import prepareDirs
import config



def main(config):
	
	prepareDirs(config)


	train_loader = train_dataloader(config.train_file_path,
									config.train_labels_path,
									config.file_type,
									config.batch_size)

	val_loader = test_dataloader(config.val_file_path,
									config.val_labels_path,
									config.file_type,
									config.batch_size)

	# print(len(train_loader.dataset))
	# print(len(val_loader.dataset))
	

	model_name = config.file_type + "_model"							
	trainer=Trainer(config,train_loader,val_loader,config.model_name)
	trainer.train()
	trainer.test(val_loader)

	# del(model)
	# temp = iter(train_loader)
	# data = temp.next()
	# # print(data.size())
	# print(data[0].size())
	# print(data[1].size())


if __name__ == "__main__":

	resume_flag = False
	debug = True
	file_type = "mfcc"
	model_name = "cnn"
	config_object = config.config_class(resume_flag,debug,file_type,model_name)

	main(config_object)


