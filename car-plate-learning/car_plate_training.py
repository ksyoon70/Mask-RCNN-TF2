import os
import os, shutil
import xml.etree
from numpy import zeros, asarray
import sys
import numpy as np
import skimage.color
import skimage.io
import skimage.transform

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import mrcnn.utils
import mrcnn.config
import mrcnn.model
import tensorflow as tf

#GPU 사용시 풀어 놓을 것
if tf.config.list_physical_devices('GPU') :
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
	config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
	
class CarPlateDataset(mrcnn.utils.Dataset):

	def __init__(self, class_map=None):
		super(CarPlateDataset,self).__init__(class_map=class_map)

	def load_dataset(self, dataset_dir, is_train=True):
		self.add_class("dataset", 1, "car")
		self.add_class("dataset", 2, "plate")
		self.add_class("dataset", 3, "kangaroo")

		images_dir = os.path.join(dataset_dir,'images') #dataset_dir + '/images/'
		#디렉토리가 없으면 생성한다.
		if not os.path.isdir(images_dir):
			os.mkdir(images_dir)
		
		annotations_dir = os.path.join(dataset_dir,'annots') #dataset_dir + '/annots/'
		if not os.path.isdir(annotations_dir):
			os.mkdir(annotations_dir)

		image_files_cnt = len(os.listdir(images_dir))
		ann_files_cnt = len(os.listdir(annotations_dir))

		if image_files_cnt == ann_files_cnt  and (image_files_cnt != 0 or ann_files_cnt !=0):
			image_filenames = os.listdir(images_dir)
			for i, filename in enumerate(image_filenames):
				image_id  = filename #image_id = i
				img_path = os.path.join(images_dir,filename)
				name = filename[:-4]
				ann_path = os.path.join(annotations_dir,name + '.xml')
				self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
			
			return True

		else:
			print("영상 dataset 파일 오류: image files coiunt = {} annotation files count = {}".format(image_files_cnt,ann_files_cnt))
			return False

		

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		tree = xml.etree.ElementTree.parse(filename)

		root = tree.getroot()

		boxes = list()
		cls_names = list()
		for obj in root.findall('.//object'):
			box = obj.find('bndbox')
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
			cls_name = obj.find('name').text
			cls_names.append(cls_name)

		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		depth = int(root.find('.//size/depth').text)
		return boxes, cls_names, width, height, depth

	# load the masks for an image
	
	def load_mask(self, image_id):
		info = self.image_info[image_id]
		path = info['annotation']
		boxes, cls_names, w, h, ch = self.extract_boxes(path)
		masks = zeros([h, w, len(boxes)], dtype='uint8')

		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			#class_ids.append(self.class_names.index('kangaroo'))
			class_ids.append(self.class_names.index(cls_names[i])) # car plate2종류가 있으므로 약간 수정
		return masks, asarray(class_ids, dtype='int32')
	
	"""
	def load_mask(self, image_id):
		#Generate instance masks for an image.
		#Returns:
		#masks: A bool array of shape [height, width, instance count] with
		#	one mask per instance.
		#class_ids: a 1D array of class IDs of the instance masks.
		
		info = self.image_info[image_id]
		# Get mask directory from image path
		mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

		# Read mask files from .png image
		mask = []
		for f in next(os.walk(mask_dir))[2]:
			if f.endswith(".png"):
				m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
				if m.shape[0] != self.image_info["height"] or m.shape[1] != self.image_info["width"]:
					m = np.ones([self.image_info["height"], self.image_info["width"]], dtype=bool)
				mask.append(m)
		mask = np.stack(mask, axis=-1)
		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID, we return an array of ones
		return mask, np.ones([mask.shape[-1]], dtype=np.int32)
	"""
	"""
	def load_mask(self, image_id):
	
	#Generate instance masks for an image.
	#Returns:
	#masks: A bool array of shape [height, width, instance count] with
	#	one mask per instance.
	#class_ids: a 1D array of class IDs of the instance masks.
	# If not a balloon dataset image, delegate to parent class.
	image_info = self.image_info[image_id]
	if image_info["source"] != "balloon":
		return super(self.__class__, self).load_mask(image_id)

	# Convert polygons to a bitmap mask of shape
	# [height, width, instance_count]
	info = self.image_info[image_id]
	mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
					dtype=np.uint8)
	for i, p in enumerate(info["polygons"]):
		# Get indexes of pixels inside the polygon and set them to 1
		rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
		mask[rr, cc, i] = 1

	# Return mask, and array of class IDs of each instance. Since we have
	# one class ID only, we return an array of 1s
	return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
	"""

	def get_class_len(self):
		return len(self.class_info)

	def get_image_len(self):
		return len(self.image_info)

class CarPlateConfig(mrcnn.config.Config):

	NUM_CLASSES = 3
	NAME = "carplate_cfg"
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	STEPS_PER_EPOCH = 130
	EPOCHS = 10  #epochs 설정

	def __init__(self, dataset) :
		#변수 초기화를 먼저 해준후 super class를 호출한다. by 윤경섭 
		CarPlateConfig.NUM_CLASSES = dataset.get_class_len()  # train set에서의 class 갯수
		CarPlateConfig.STEPS_PER_EPOCH = dataset.get_image_len() # train set에서의 영상 갯수
		super(CarPlateConfig, self).__init__()

# prepare train set
train_set = CarPlateDataset()
car_plate_dir = os.path.join(os.path.dirname(__file__),'car-plate')
if not os.path.isdir(car_plate_dir):
	os.mkdir(car_plate_dir)

train_dataset_dir = os.path.join(car_plate_dir,'training')
#디렉토리가 없으면 만든다.
if not os.path.isdir(train_dataset_dir):
	os.mkdir(train_dataset_dir)

if train_set.load_dataset(dataset_dir=train_dataset_dir, is_train=True) == True :
	train_set.prepare()

else :
    # 데이터셋 준비에 이상이 있으면 진행을 종료 시킨다.
    quit()

# prepare test/val set
valid_dataset = CarPlateDataset()
valid_dataset_dir = os.path.join(car_plate_dir,'valid')
#디렉토리가 없으면 만든다.
if not os.path.isdir(valid_dataset_dir):
	os.mkdir(valid_dataset_dir)
if valid_dataset.load_dataset(dataset_dir=valid_dataset_dir, is_train=False) == True:
	valid_dataset.prepare()
else :
    # 데이터셋 준비에 이상이 있으면 진행을 종료 시킨다.
    quit()

# prepare config
car_plate_config = CarPlateConfig(train_set)
model_dir = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'mrcnn')
# define the model
model = mrcnn.model.MaskRCNN(mode='training', 
							 model_dir=model_dir, 
							 config=car_plate_config)

mrcnn_weight_dir = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))),'mask_rcnn_coco.h5')
model.load_weights(filepath=mrcnn_weight_dir, 
				   by_name=True, 
				   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_set, 
			val_dataset=valid_dataset, 
			learning_rate=car_plate_config.LEARNING_RATE, 
			epochs=car_plate_config.EPOCHS, 
			layers='heads')

model_path = 'CarPlate_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
