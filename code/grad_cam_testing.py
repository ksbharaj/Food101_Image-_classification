# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 19:04:10 2022

@author: bwoodman
"""

# https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
import torch
import matplotlib.pyplot as plt
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
from collections import OrderedDict
import numpy as np
import argparse
import os
import torch.nn as nn
import pandas as pd
from torchvision import datasets, transforms, models
import torch.utils.data as data
import PIL
i=0



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load(r'../checkpoints/food-101_best_augmented.pt')['model']
model.eval()

# Set the model on Eval Mode
#model.eval()
model = model.to(device)

train_df = pd.read_hdf(r'../data-frames/train_df.h5')
val_df = pd.read_hdf(r'../data-frames/test_df.h5')
test_df = pd.read_hdf(r'../data-frames/test_df.h5')

col_names = list(train_df.columns.values)
ing_names = col_names[:-3]
targets = ing_names

image_transform = transforms.Compose([transforms.Resize((384,384)),
                                       transforms.ToTensor()])



class CustomDataLoader(data.Dataset):
    ''' Data wrapper for pytorch's data loader function '''
    def __init__(self, image_df):
        self.dataset = image_df

    def __getitem__(self, index):
        c_row = self.dataset.iloc[index]
        target_arr = []
        for item in c_row[targets].values:
            target_arr.append(item)
        #print(target_arr)
        image_path, target = c_row['path'], torch.from_numpy(np.array(target_arr)).float()  #image and target
        #read as rgb image, resize and convert to range 0 to 1
        
        
        image = cv2.imread(image_path, 1)
        print(image_path)
        image = PIL.Image.fromarray(image)
        image = image_transform(image) 
        return image, target

    def __len__(self):
        return self.dataset.shape[0]
    
train_dataset = CustomDataLoader(train_df)

val_dataset = CustomDataLoader(val_df)

test_dataset = CustomDataLoader(test_df)



class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            #print('name=',name)
            #print('x.size()=',x.size())
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            #print('outputs.size()=',x.size())
        #print('len(outputs)',len(outputs))
        return outputs, x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers,use_cuda):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = True
	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output  = self.feature_extractor(x)
		output = output.view(output.size(0), -1)
		model = torch.load(r'../checkpoints/food-101_best_augmented.pt')['model']
		
       
        
		model = model.to(device)
		if self.cuda:
			output = output.cuda()
			output = model.fc(output).cuda()
		else:
			output = model.fc(output)
		return target_activations, output


def show_cam_on_image(img, mask,name):
	heatmap = cv2.applyColorMap(np.uint8(255* mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("./gc_images/gradcam_{}.jpg".format(name), np.uint8(255 * cam))
    
class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)


    
	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()
		one_hot.backward(retain_graph=True)##
		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		target = features[-1]
		target = target.cpu().data.numpy()[0, :]
		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)
		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (384, 384))
        
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
        
		return cam
    

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='../food-101/images/',
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = True#args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")

	return args
   
args = get_args()
model = torch.load(r'../checkpoints/food-101_best_augmented.pt')['model']
model_o = torch.load(r'../checkpoints/food-101_best_augmented.pt')['model']
del model.fc


#Evaluating Various Aspects of the Test Set based on anommalies found in the visualizations
    #cheese_plate 17 all identical ingredients
    #beet_salad 5 all identical ingredients
    #cilantro - 'ceviche', 'chicken_curry', 'guacamole', 'huevos_rancheros', 'miso_soup', 'nachos', 'pad_thai', 'pho', 'tacos', 'tuna_tartare'
    #beef - 'beef_carpaccio', 'french_onion_soup', 'hamburger', 'nachos', 'pho', 'spaghetti_bolognese', 'tacos'
    #steak - 'beef_tartare', 'bibimbap', 'steak'
    # train_df[train_df['class_id'] == '5'].sum()
    #pd.Series(train_df[train_df['cilantro'] == 1]['target'].values).unique()
    # pd.Series(train_df[train_df['beef'] == 1]['target'].values).unique()
    # pd.Series(train_df[train_df['steak'] == 1]['target'].values).unique()
    # df = train_df[train_df['class_id'] == '38'].sum()
    # df = train_df[train_df['class_id'].isin( ['15', '18', '51', '56', '64', '66', '70', '75', '96', '99'])].sum()
    # test_df[test_df['path'].apply(lambda x: '2025645' in str(x) )]['path']
    # test_df.loc[3615, 'path']


model = model.to(device)
try:
    model.fc
except:
    pass
grad_cam = GradCam(model , target_layer_names = ["layer4"], use_cuda=True)	
   
x=os.walk(args.image_path)
image = []
image_name = []
count =1
for path, directories, files in os.walk(r'../food-101/images/'):
    for file in files:
        count = count + 1
	    #print('found %s' % os.path.join(path, file))
        if count >20:
            image.append(cv2.imread(path+'/'+file,1))
            image_name.append('{}{}'.format(path.split('/')[-1], file))
            count = 0
            
                  
count = 0

#produce gradcam for 50%+ likelihood images
from torchvision import transforms
for img in image:
    image_transform = transforms.Compose([transforms.Resize((384,384)),
                                       transforms.ToTensor()])
        
    print(image_name[count].split('\\')[-1] )
    # PIL.Image.fromarray(img)
    # C:/Users/bwood/food_101/food-101/images/fish_and_chips/2025645.jpg
    #img = cv2.imread(r'C:/Users/bwood/OneDrive/Documents/Data Science/CS 7643/Project/cilantro_0003.jpg',1) #C:\Users\bwood\OneDrive\Documents\Data Science\CS 7643\Project
    #img = cv2.imread(r'C:/Users/bwood/food_101/food-101/images/fish_and_chips/2025645.jpg',1) #C:\Users\bwood\OneDrive\Documents\Data Science\CS 7643\Project
   
    input = image_transform(PIL.Image.fromarray(img)) 
    input = input.reshape(1, 3, 384, 384)
    input.required_grad = True
    input = input.to(device)

    

    model = model.to(device)
    output = model_o(input)
    preds = torch.sigmoid(output).data >= 0.5
    pred_list = test_df.iloc[:, list(list(preds.data.cpu())[0])+ [False, False, False]].columns
        
    try:
    	del model.fc
    except:
        pass
        
    for ing in pred_list:
        
        # ing = 'apple'
        ing_index = test_df.columns.get_loc(ing)
    # for ing_index in range(273):
        grad_cam = GradCam(model , target_layer_names = ["layer4"], use_cuda=True)	
        mask = grad_cam(input, ing_index)
        i=i+1 
        #show_cam_on_image(np.float32(cv2.resize(img, (384, 384))) / 255, mask, test_df.columns[ing_index] + '_' + image_name[count].split('\\')[-1])
        
        
        show_cam_on_image(np.float32(cv2.resize(img, (384, 384))) / 255, mask,  image_name[count].split('\\')[-1]  + '_' + test_df.columns[ing_index])
        
        
    count = count +1
    



class GradCam_not_normal:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)


    
	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()
		one_hot.backward(retain_graph=True)##
		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		target = features[-1]
		target = target.cpu().data.numpy()[0, :]
		weights = np.mean(grads_val, axis = (2, 3))[0, :]
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)
		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (384, 384))
        
		cam = cam - np.min(cam)
# 		cam = cam / np.max(cam)
        
		return cam

#produce grad cam for low ranked ingredients
preds_low = torch.sigmoid(output).data <= 0.00000006
test_df.iloc[:, list(list(preds_low.data.cpu())[0])+ [False, False, False]].columns
pred_list_low = test_df.iloc[:, list(list(preds_low.data.cpu())[0])+ [False, False, False]].columns
for ing in pred_list:
    
    #ing = 'pickle'
    ing_index = test_df.columns.get_loc(ing)
    # for ing_index in range(273):
    grad_cam = GradCam_not_normal(model , target_layer_names = ["layer4"], use_cuda=True)	
    mask = grad_cam(input, ing_index)
    i=i+1 
    #show_cam_on_image(np.float32(cv2.resize(img, (384, 384))) / 255, mask, test_df.columns[ing_index] + '_' + image_name[count].split('\\')[-1])
    
    
    show_cam_on_image(np.float32(cv2.resize(img, (384, 384))) / 255, mask,  image_name[count].split('\\')[-1]  + '_' + test_df.columns[ing_index])
    
    
count = count +1
