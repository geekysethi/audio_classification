import numpy as np
import shutil
import os
import time
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn import preprocessing



def prepareDirs(config):

	print('==> Preparing data..')
	if not os.path.exists(config.output_model_dir):
		os.makedirs(config.output_model_dir)

	if not os.path.exists(config.output_model_dir+"/"+config.model_name):
		os.makedirs(config.output_model_dir+"/"+config.model_name)

	if not os.path.exists(config.output_model_dir+"/"+config.model_name+"/"+config.file_type):
		os.makedirs(config.output_model_dir+"/"+config.model_name+"/"+config.file_type)

	if not os.path.exists(config.output_model_dir+"/"+config.model_name+"/"+config.file_type +"/data"):
		os.makedirs(config.output_model_dir+"/"+config.model_name+"/"+config.file_type + "/data")


def prepareData(data):
	finalData=[]
	for i in data:

		try:
			for j in i:
				finalData.append(j)
		except:
			print("ERROR")
	finalData=np.array(finalData)
	# print(finalData.shape)
	return finalData



_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
	global last_time, begin_time
	if current == 0:
		begin_time = time.time()  # Reset for new bar.

	cur_len = int(TOTAL_BAR_LENGTH*current/total)
	rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

	sys.stdout.write(' [')
	for i in range(cur_len):
		sys.stdout.write('=')
	sys.stdout.write('>')
	for i in range(rest_len):
		sys.stdout.write('.')
	sys.stdout.write(']')

	cur_time = time.time()
	step_time = cur_time - last_time
	last_time = cur_time
	tot_time = cur_time - begin_time

	L = []
	L.append('  Step: %s' % format_time(step_time))
	L.append(' | Tot: %s' % format_time(tot_time))
	if msg:
		L.append(' | ' + msg)

	msg = ''.join(L)
	sys.stdout.write(msg)
	for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
		sys.stdout.write(' ')

	# Go back to the center of the bar.
	for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
		sys.stdout.write('\b')
	sys.stdout.write(' %d/%d ' % (current+1, total))

	if current < total-1:
		sys.stdout.write('\r')
	else:
		sys.stdout.write('\n')
	sys.stdout.flush()


def format_time(seconds):
	days = int(seconds / 3600/24)
	seconds = seconds - days*3600*24
	hours = int(seconds / 3600)
	seconds = seconds - hours*3600
	minutes = int(seconds / 60)
	seconds = seconds - minutes*60
	secondsf = int(seconds)
	seconds = seconds - secondsf
	millis = int(seconds*1000)

	f = ''
	i = 1
	if days > 0:
		f += str(days) + 'D'
		i += 1
	if hours > 0 and i <= 2:
		f += str(hours) + 'h'
		i += 1
	if minutes > 0 and i <= 2:
		f += str(minutes) + 'm'
		i += 1
	if secondsf > 0 and i <= 2:
		f += str(secondsf) + 's'
		i += 1
	if millis > 0 and i <= 2:
		f += str(millis) + 'ms'
		i += 1
	if f == '':
		f = '0ms'
	return f



def normalized_confusion_matrix(true_labels,predict_labels,classes_list,save_path):

	cmap=plt.cm.Blues

	cm = confusion_matrix(true_labels, predict_labels)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print(cm)
	cm =np.round(cm,2)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title("confusion matrix")
	plt.colorbar()
	tick_marks = np.arange(len(classes_list))
	plt.xticks(tick_marks, classes_list, rotation=45)
	plt.yticks(tick_marks, classes_list)

	thresh = cm.max() / 2.
	for i in range (cm.shape[0]):
		for j in range (cm.shape[1]):
			plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(save_path + "confusion_matrix.png",dpi = 300)

	return None


def one_hot(labels):
	
	label_binarizer = preprocessing.LabelBinarizer()
	label_binarizer.fit(range(max(labels)+1))
	b = label_binarizer.transform(labels)

	return b

def plot_roc_curve(true_labels,pred_labels,classes_list,save_path):



	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in classes_list:
		fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_labels[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])



	plt.figure()
	for i in classes_list:
		plt.plot(fpr[i], tpr[i], label='Class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC CURVE')
	plt.legend(loc="lower right")
	plt.savefig(save_path+"roc_curve.png",dpi = 300)
	