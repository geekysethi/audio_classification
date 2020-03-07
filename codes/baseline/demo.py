from trainer import Trainer
from data_loader import train_dataloader, test_dataloader
import config



def main(config):
	

	train_loader = train_dataloader(config.train_file_path,
									config.train_labels_path,
									config.file_type,
									config.batch_size)

	val_loader = test_dataloader(config.val_file_path,
									config.val_labels_path,
									config.file_type,
									config.batch_size)

	

	trainer=Trainer(config,train_loader,val_loader,config.model_name)
	trainer.test(val_loader)



if __name__ == "__main__":

	resume_flag = True
	debug = True
	file_type = "mfcc"
	model_name = "cnn"
	config_object = config.config_class(resume_flag,debug,file_type,model_name)

	main(config_object)


