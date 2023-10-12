import numpy as np

def get_confusion_matrix(label, pred, num_class, ignore=255):
	"""
	Calcute the confusion matrix by given label and pred
	"""
	output = pred.cpu().numpy().transpose(0, 2, 3, 1)
	#mask = label.cpu().numpy().transpose(0, 2, 3, 1)
	seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
	#seg_gt = np.asarray(np.argmax(mask, axis=3), dtype=np.int)
	seg_gt = label.cpu().numpy()

	ignore_index = seg_gt != ignore
	seg_gt = seg_gt[ignore_index]
	seg_pred = seg_pred[ignore_index]

	index = (seg_gt * num_class + seg_pred).astype('int32')
	label_count = np.bincount(index)
	confusion_matrix = np.zeros((num_class, num_class))

	for i_label in range(num_class):
		for i_pred in range(num_class):
			cur_index = i_label * num_class + i_pred
			if cur_index < len(label_count):
				confusion_matrix[i_label,
								 i_pred] = label_count[cur_index]
	return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters,cur_iters, power=0.9):
	lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
	optimizer.param_groups[0]['lr'] = lr
	return lr

def get_data_class(data_name):

	if "camvid" in data_name:

		CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
				   'tree', 'signsymbol', 'fence', 'car', 
				   'pedestrian', 'bicyclist']
	elif data_name=='voc':

		CLASSES= ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
				  'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 
				  'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
	elif "city" in data_name:

		CLASSES=['road','sidewalk','building','wall','fence',
				'pole','traffic_light','traffic_sign','vegetation','terrain','sky',
				 'person','rider','car','truck','bus','train',
				'motocycle','bicycle']

	return len(CLASSES)

