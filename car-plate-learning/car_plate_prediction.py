import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import skimage.transform

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

#CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
CLASS_NAMES = ['BG','car','plate','kangaroo']
class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="CarPlate_mask_rcnn_trained.h5", 
                   by_name=True)

car_plate_dir = os.path.join(os.path.dirname(__file__),'car-plate')
if not os.path.isdir(car_plate_dir):
	os.mkdir(car_plate_dir)

test_dataset_dir = os.path.join(car_plate_dir,'test')
#디렉토리가 없으면 만든다.
if not os.path.isdir(test_dataset_dir):
	os.mkdir(test_dataset_dir)

images_dir = os.path.join(test_dataset_dir,'images') #dataset_dir + '/images/'
#디렉토리가 없으면 만든다.
if not os.path.isdir(images_dir):
	os.mkdir(images_dir)

for filename in os.listdir(images_dir):
	# load the input image, convert it from BGR to RGB channel
	image_path = os.path.join(images_dir,filename)
	#image_path = images_dir + filename
	#image_path = image_path.replace('\\','//')
	#image = plt.imread(image_path)

	if os.path.exists(image_path) :
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# Perform a forward pass of the network to obtain the results
		r = model.detect([image], verbose=0)
		
		# Get the results for the first image.
		r = r[0]
		
		# Visualize the detected objects.
		mrcnn.visualize.display_instances(image=image, 
										boxes=r['rois'], 
										masks=r['masks'], 
										class_ids=r['class_ids'], 
										class_names=CLASS_NAMES, 
										scores=r['scores'])
	else :
		print('image path : {} is not exist!'.format(image_path))
	
	#image = skimage.io.imread(image_path)
	#if image.ndim != 3:
	#	image = skimage.color.gray2rgb(image)

	
