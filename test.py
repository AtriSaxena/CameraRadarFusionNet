import keras
import keras.preprocessing.image
import tensorflow as tf
import progressbar
import cv2
import numpy as np
import pprint
import time
import multiprocessing

from crfnet.model import architectures
from crfnet.model.architectures.retinanet import retinanet_bbox
from crfnet.utils.config import get_config
from crfnet.utils.keras_version import check_keras_version
from crfnet.data_processing.fusion.fusion_projection_lines import create_imagep_visualization
from crfnet.utils.anchor_parameters import AnchorParameters
from crfnet.utils.colors import tum_colors
from crfnet.data_processing.generator.crf_main_generator import create_generators


backbone = architectures.backbone('vgg-max-fpn')
model = keras.models.load_model('/kaggle/working/CameraRadarFusionNet/crfnet/saved_models/crf_net.h5', custom_objects=backbone.custom_objects)


anchor_params = AnchorParameters.small
num_anchors = AnchorParameters.small.num_anchors()

prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params, class_specific_filter=False)

inputs = np.load('/kaggle/input/sources/input_ti_data_04_08_01.npy')
inputs.shape

boxes, scores, labels = prediction_model.predict_on_batch(inputs)[:3]

with open('boxes.npy','wb') as f:
    np.save(f, boxes)
    
with open('scores.npy','wb') as f:
    np.save(f, scores)
    
with open('labels.npy','wb') as f:
    np.save(f, labels)

selection = np.where(scores > 0.4)[1]
boxes = boxes[:,selection,:]
scores = scores[:,selection]
labels = labels[:,selection]
predictions = [boxes, scores, labels] 

print(predictions)