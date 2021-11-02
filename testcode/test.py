import os
import os, shutil
import xml.etree
from numpy import zeros, asarray
import sys
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
import json			#json 파일을 읽기 위하여 추가 by 윤경섭



filename = os.path.join(os.path.dirname(__file__),'095451466828260_10오8116.json')
class CarPlateDataset():

	def __init__(self, class_map=None):
		self._image_ids = []
		self.image_info = []
		# Background is always the first class
		self.class_info = [{"source": "", "id": 0, "name": "BG"}]
		self.source_class_ids = {}

		self.add_class("dataset", 1, "car")
		self.add_class("dataset", 2, "plate")
		self.add_class("dataset", 3, "kangaroo")
		self.add_image("dataset",image_id = '095451466828260_10오8116.jpg',path=filename)

		def clean_name(name):
			"""Returns a shorter version of object names for cleaner display."""
			return ",".join(name.split(",")[:1])

		self.class_names = [clean_name(c["name"]) for c in self.class_info]

	def add_class(self, source, class_id, class_name):
		assert "." not in source, "Source name cannot contain a dot"
		# Does the class exist already?
		for info in self.class_info:
			if info['source'] == source and info["id"] == class_id:
				# source.class_id combination already available, skip
				return
		# Add the class
		self.class_info.append({
			"source": source,
			"id": class_id,
			"name": class_name,
		})
	
	def add_image(self, source, image_id, path, **kwargs):
		image_info = {
			"id": image_id,
			"source": source,
			"path": path,
		}
		image_info.update(kwargs)
		self.image_info.append(image_info)

	def extract_boxes(self,filename):
		boxes = list()
		cls_names = list()


		with open(filename, 'r',encoding="UTF-8") as f:
			json_data = json.load(f)
			for item, shape in enumerate(json_data['shapes']):
				cls_names.append(shape['label'])
				points = shape['points']
				arr = np.array(points)
				""""
				xmin = np.min(arr[:,0])
				ymin = np.min(arr[:,1])
				xmax = np.max(arr[:,0])
				ymax = np.max(arr[:,1])
				print('class name is :{} box is {}, {}, {}, {}'.format(cls_name,xmin,ymin,xmax,ymax))
				"""
				boxes.append(points)

				width = json_data['imageWidth']
				height = json_data['imageHeight']
				depth = 3 # labelme에서는 채널을 저장안하므로 기본 3을 써준다. 어짜피 읽는 쪽에서 안쓴다. by 윤경섭

		return boxes, cls_names, width, height, depth

	def load_mask(self,image_id):
		info = self.image_info[image_id]
		path = info['path']
		boxes, cls_names, w, h, ch = self.extract_boxes(path)
		masks = zeros([h, w, len(boxes)], dtype='uint8')

		class_ids = list()

			#Generate instance masks for an image.
			#Returns:
			#masks: A bool array of shape [height, width, instance count] with
			#	one mask per instance.
			#class_ids: a 1D array of class IDs of the instance masks.
			# If not a balloon dataset image, delegate to parent class.
			#image_info = self.image_info[image_id]
			#if image_info["source"] != "balloon":
			#	return super(self.__class__, self).load_mask(image_id)

			# Convert polygons to a bitmap mask of shape
			# [height, width, instance_count]
			#info = self.image_info[image_id]
		for i, p in enumerate(boxes):
			# Get indexes of pixels inside the polygon and set them to 1
			xpt = [ i[0] for i in p]
			ypt = [ i[1] for i in p]
			rr, cc = skimage.draw.polygon(xpt, ypt)
			masks[cc, rr, i] = 1
			class_ids.append(self.class_names.index(cls_names[i]))
			# Return mask, and array of class IDs of each instance. Since we have
			# one class ID only, we return an array of 1s
			#return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
		return masks, asarray(class_ids, dtype='int32')

train_set = CarPlateDataset()

train_set.load_mask(image_id = 0)
