from torchvision import models
import torch.nn as nn

def initialize_model(modelName,usePretrained):
	model_ft = None

	if modelName == "resnet":
		""" Resnet18
		"""
		print("MODEL RESNET-18")
		model_ft = models.resnet18(pretrained=usePretrained)
		num_ftrs = model_ft.fc.in_features
		model_ft.fc = nn.Linear(num_ftrs, 2)
		new_classifier = nn.Sequential(*list(model_ft.children())[:-1])
		model_ft = new_classifier
		
	
		
	
	elif modelName == "alexnet":
		""" Alexnet
		"""
		model_ft = models.alexnet(pretrained=usePretrained)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,2)
		new_classifier = nn.Sequential(*list(model_ft.classifier.children())[:-1])
		model_ft.classifier = new_classifier


	elif modelName == "vgg":

		print("MODEL VGG-NET-16 WITHOUT BATCH NORMALIZATION")
		model_ft = models.vgg16(pretrained=usePretrained)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,2)

		new_classifier = nn.Sequential(*list(model_ft.classifier.children())[:-1])
		model_ft.classifier = new_classifier
		
	elif modelName == "vgg_bn":
		""" VGG11_bn
		"""
		print("MODEL VGG-NET-16 WITH BATCH NORMALIZATION")
		model_ft = models.vgg16_bn(pretrained=usePretrained)
		num_ftrs = model_ft.classifier[6].in_features
		model_ft.classifier[6] = nn.Linear(num_ftrs,2)
		new_classifier = nn.Sequential(*list(model_ft.classifier.children())[:-1])
		model_ft.classifier = new_classifier
		
		

	elif modelName == "squeezenet":
		""" Squeezenet
		"""
		model_ft = models.squeezenet1_1(pretrained=usePretrained)
		model_ft.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1,1), stride=(1,1))
		model_ft.num_classes = 2

		new_classifier = nn.Sequential(*list(model_ft.classifier.children())[:-4])
		model_ft.classifier = new_classifier


		

	

	else:
		print("Invalid model name, exiting...")
	return model_ft
	

	