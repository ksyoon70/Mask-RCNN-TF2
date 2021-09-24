import os
import os, shutil
import xml.etree
from numpy import zeros, asarray
import sys

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

	def load_dataset(self, dataset_dir, is_train=True):
		self.add_class("dataset", 1, "car")
		self.add_class("dataset", 2, "plate")

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
				image_id = i
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

class CarPlateConfig(mrcnn.config.Config):
	NAME = "carplate_cfg"

	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	
	NUM_CLASSES = 3

	STEPS_PER_EPOCH = 131

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
car_plate_config = CarPlateConfig()
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
			epochs=1, 
			layers='heads')

model_path = 'CarPlate_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
