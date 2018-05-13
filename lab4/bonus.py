import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, utils
import os
import math
from torch.autograd import Variable
import tensorflow as tf
import cv2
from logger import Logger

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class CNN(nn.Module):

	def __init__(self, block, layers, num_classes=15):
		self.inplanes = 64
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AvgPool2d(7, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x
def resnet20():
	print(20)
	return CNN(BasicBlock, [3,3,3])

def resnet50():
	#print(56)
	return CNN(Bottleneck, [3, 4, 6, 3])

def resnet110():
	print(110)
	return CNN(BasicBlock, [18,18,18])

def train(epoch):
	global train_iter
	global logger
	global _batch_size
	print('Current Epoch : %d' % epoch)
	cnn.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(data_loader):
		onehot = torch.zeros([_batch_size, 15], dtype=torch.long)
		for i in range(_batch_size):
			for j in range(15):
				if j == targets[i]:
					onehot[i][j] = 1.
		label = onehot
		inputs, label = inputs.cuda(), label.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(label)
		outputs = cnn(inputs)
		loss = loss_function(outputs, torch.max(label, 1)[1])
		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
		_, predicted = torch.max(outputs.data, 1)
		total += label.size(0)
		correct += predicted.eq(torch.max(label, 1)[1]).cpu().sum()
		info = {'Train_Error%' : 100 - 100 * correct/total}
		for tag, value in info.items():
			logger_cv.scalar_summary(tag, value, train_iter)
		train_iter += 1
		print('Current Train : Loss: %.3f | Acc: %.3f%% (%d/%d)' % (loss.data[0], 100.*predicted.eq(torch.max(label, 1)[1]).cpu().sum()/label.size(0), correct, total))
	print('Train : Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
	global best_acc
	global test_iter
	global logger
	cnn.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(testloader):
		onehot = torch.zeros([_batch_size, 15], dtype=torch.long)
		for i in range(_batch_size):
			for j in range(15):
				if j == targets[i]:
					onehot[i][j] = 1.
		label = onehot
		inputs, label = inputs.cuda(), label.cuda()
		inputs, label = Variable(inputs), Variable(label)
		outputs = cnn(inputs)
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(torch.max(label, 1)[1]).cpu().sum()
		#test_error.append(round(correct/total, 3))
		info = {'Test_Error%' : 100 - 100 * correct/total}
		for tag, value in info.items():
			logger_cv.scalar_summary(tag, value, test_iter)
		test_iter += 1
	if best_acc < 100. * correct/total:
		best_acc = 100. * correct/total
		torch.save(cnn, './model56')		

	print('Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
best_acc = 0.
LR = 0.1
test_iter = 0
train_iter = 0
_batch_size = 10
cnn = resnet50()
cnn.cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
transform_train = transforms.Compose([
	transforms.RandomCrop(244, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
])
img_train = torchvision.datasets.ImageFolder('./train',
                                            transform=transforms.Compose([
                                                transforms.CenterCrop(224),
                                                transforms.Resize(size=220),
                                                transforms.ToTensor()])
                                            )
data_loader = torch.utils.data.DataLoader(img_train, batch_size=_batch_size,shuffle=True)
img_test = torchvision.datasets.ImageFolder('./test',
                                            transform=transforms.Compose([
                                                transforms.CenterCrop(224),
                                                transforms.Resize(size=220),
                                                transforms.ToTensor()])
                                            )
testloader = torch.utils.data.DataLoader(img_test, batch_size=_batch_size,shuffle=True)
logger_cv = Logger('./logs_cv')
def main():
	global LR
	for epoch in range(1, 165):
		if epoch == 81 or epoch == 122:
			LR /= 10
		train(epoch)
		test(epoch)	
	

if __name__ == "__main__":
	main()
