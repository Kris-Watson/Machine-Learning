# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:37:56 2022

@author: General
"""

# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn
# Allows us to transform data
import torchvision.transforms as transforms
# Allows us to download the dataset
import torchvision.datasets as dsets
# Used to graph data and loss curves
import matplotlib.pylab as plt
# Allows us to use arrays to manipulate and store data
import numpy as np
import copy
from PIL import Image
import os

# Python program to show time by perf_counter()
from time import perf_counter

import torchvision.models as models
from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.model_selection import train_test_split as tts



"""
Set our device to use gpu if available.
Later called by <object>.to(device) to send a neural network or pytorch..
tensor to gpu memory and run any calculations on the gpu.
Comparing data from RAM and GPU memory will throw an exception.
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the device type is", device)




# Plots our accuracy and loss for each iteration
def plot_stuff(COST,ACC):
	fig, ax1 = plt.subplots()
	color = 'tab:red'
	ax1.plot(COST, color = color)
	ax1.set_xlabel('Iteration', color = color)
	ax1.set_ylabel('total loss', color = color)
	ax1.tick_params(axis = 'y', color = color)

	ax2 = ax1.twinx()
	color = 'tab:blue'
	ax2.set_ylabel('accuracy', color = color)  # we already handled the x-label with ax1
	ax2.plot(ACC, color = color)
	ax2.tick_params(axis = 'y', color = color)
	fig.tight_layout()  # otherwise the right y-label is slightly clipped

	plt.show()




# Define a transform for our images to be comparable in the nn
IMAGE_SIZE = 500
composed = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
							   transforms.ToTensor()])

# Load our images from Data/stop and Data/not_stop
"""
We use os.listdir to extract filenames.
Images are not the same type so we use .convert('RGB') to keep images consistent
"""

def load_images():
	# loop over the input images
	dataset = []

	# Label numbers for our images
	stop_label = 1
	not_stop_label = 0

	# Check for images in stop folder and label them 1
	for filename in os.listdir('Data/stop'):
		image = Image.open(f'Data/stop/{filename}')
		if not image.mode == 'RGB':
			image = image.convert('RGB')
		# Set hotdog to label 1 and not_hotdog to label 0
		image = composed(image)
		dataset.append([image, stop_label])

	# Check for images in not_stop folder and label them 0
	for filename in os.listdir('Data/not_stop'):
		image = Image.open(f'Data/not_stop/{filename}')
		if not image.mode == 'RGB':
			image = image.convert('RGB')
		# Set hotdog to label 1 and not_hotdog to label 0
		image = composed(image)
		dataset.append([image, not_stop_label])

	return dataset


# Function that trains the model with data using defined hyperparameters
def train_model(model, train_loader,validation_loader, criterion, optimizer, n_epochs,print_=True):
	loss_list = []
	accuracy_list = []
	correct = 0
	#global:val_set
	n_test = len(val_set)
	accuracy_best=0
	best_model_wts = copy.deepcopy(model.state_dict())


	#for epoch in tdqm(range(n_epochs)):
	for epoch in range(n_epochs):
		loss_sublist = []
		# Loop through the data in loader

		for x, y in train_loader:
			# Use gpu for images
			x, y=x.to(device), y.to(device)
			# Set model in training mode
			model.train()
			# Calculate a prediction
			z = model(x)
			# Compare predication with actual value
			loss = criterion(z, y)
			loss_sublist.append(loss.data.item())
			# Propogates loss backwards through each layer
			loss.backward()
			# Use optimizer to change w and b values based on loss
			optimizer.step()

			optimizer.zero_grad()
		print("epoch {} done".format(epoch))

		scheduler.step()
		loss_list.append(np.mean(loss_sublist))
		correct = 0

		# Check model prediction accuracy
		# Predicts an output and checks the most probable label vs actual label
		for x_test, y_test in validation_loader:
			x_test, y_test=x_test.to(device), y_test.to(device)
			model.eval()
			z = model(x_test)
			_, yhat = torch.max(z.data, 1)
			correct += (yhat == y_test).sum().item()
		accuracy = correct / n_test
		accuracy_list.append(accuracy)
		if accuracy>accuracy_best:
			accuracy_best=accuracy
			best_model_wts = copy.deepcopy(model.state_dict())


		if print_:
			print('learning rate',optimizer.param_groups[0]['lr'])
			print("The validaion  Cost for each epoch "+str(epoch + 1)+": "+str(np.mean(loss_sublist)))
			print("The validation accuracy for epoch "+str(epoch + 1)+": "+str(accuracy))
	model.load_state_dict(best_model_wts)
	return accuracy_list,loss_list, model





# Use a pre-defined resnet18 model as our base, later training it to fit our..
# data.
"""
Resnet18 is a model that uses residual learning to speed up training for
models with many layers by passing the value of a node (residue) to a node 2
layers down.
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#resnet
"""
model = models.resnet18(pretrained=True)


# We will only train the last layer of the network, so we set the parameter..
# requires_grad to False - the network is a fixed feature extractor.
for param in model.parameters():
	param.requires_grad = False



# Default for the last layer (model.fc) is:
# (fc): Linear(in_features=512, out_features=1000, bias=True)
# Since we only have 2 classes (stop or not_stop) we change this to:
n_classes = 2
model.fc = nn.Linear(512, n_classes)

# Use gpu for model training and predictions
model.to(device)

# Get data and split it into test and train sets
dataset = load_images()
train_set, val_set = tts(dataset,train_size=0.9)

# Build our batches:
batch_size=32
train_loader = torch.utils.data.DataLoader(dataset=train_set,
										   batch_size=batch_size,shuffle=True)
validation_loader= torch.utils.data.DataLoader(dataset=val_set ,
											   batch_size=1)


# Define our hyperparams
n_epochs=10
lr=0.000001
momentum=0.9
# lr_scheduler: For each epoch change the learning rate between base_lr and..
# max_lr. Allows to quickly train an accurate model without the need to..
# manually change learning rate.
lr_scheduler=True
base_lr=0.001
max_lr=0.01


# Our optimizer and criterion
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
if lr_scheduler:
	scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
												  base_lr=0.001,
												  max_lr=0.01,
												  step_size_up=5,
												  mode="triangular2")



# Train our model

# Start the stopwatch / counter
t1_start = perf_counter()

# Train the model and retrieve loss and accuracy values
accuracy_list,loss_list, model=train_model(model,
										   train_loader,
										   validation_loader,
										   criterion, optimizer,
										   n_epochs=n_epochs)

# Stop the stopwatch / counter
t1_stop = perf_counter()
elapsed_time = t1_stop - t1_start
print("elapsed time", elapsed_time, "s")

# Save the model to model.pt
torch.save(model.state_dict(), 'trained_model.pt')

# Plot loss and accuracy for each iteration
plot_stuff(loss_list,accuracy_list)




"""
# Load the model by creating a new resnet18 model then loading the params..
# (w and b) from our pt file in models folder
"""
n_classes = 2
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, n_classes)
model.load_state_dict(torch.load( "trained_model.pt"))
model = model.to(device)
model.eval()

# Collect all images from myImages folder
myData = []
for filename in os.listdir('myImages'):
	img_path = f'myImages/{filename}'
	image = Image.open(img_path)
	if not image.mode == 'RGB':
		image = image.convert('RGB')
	# Set hotdog to label 1 and not_hotdog to label 0
	myImage = composed(image).to(device)
	myData.append([myImage, filename])


for img, path in myData:
	# Get prediction
	z = model(img.unsqueeze(0))
	# Print label with highest probability
	myProb, myLabel = torch.max(z.data, 1)

	if myLabel.item() == 1:
		myLabel = "has a stop sign"
	elif myLabel.item() == 0:
		myLabel = "does not have a stop sign"
	print(f'{path} {myLabel}, with a probability of {myProb.item()}')
