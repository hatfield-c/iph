import torch
import numpy as np
import cv2
import random

import CONFIG

class DataLoader:
	def __init__(self):
		self.img_path = "data/in/" + CONFIG.img_name
		
		img_raw = cv2.imread(self.img_path)
		
		img_size = [720, 1280]
		
		if img_raw.shape[1] > img_raw.shape[0]:
			downratio = img_raw.shape[1] / img_size[1]
			img_size[0] = int(img_raw.shape[0] // downratio)
		else:
			downratio = img_raw.shape[0] / img_size[0]
			img_size[1] = int(img_raw.shape[1] // downratio)
			
		img_raw = cv2.resize(img_raw, (img_size[1], img_size[0]))
		
		self.img_raw = img_raw
		self.img_size = torch.FloatTensor(img_size)
		self.img_grid = self.ImageGrid(img_size[0], img_size[1])
		self.img = torch.FloatTensor(img_raw)
		self.img = torch.cat((self.img_grid, self.img), dim = 2)
		self.img = self.img / torch.FloatTensor([img_size[0], img_size[1], 255, 255, 255])
	
		#stride_0 = torch.arange(0, self.img.shape[0], 10)
		#stride_1 = torch.arange(0, self.img.shape[1], 10)
		#self.img = self.img[stride_0]
		#self.img = self.img[:, stride_1]
		#self.img = self.img.reshape(-1, 5)
		
		#eimg = cv2.Canny(img_raw, 50, 200)
		#eimg = cv2.dilate(eimg, np.ones((5, 5), np.uint8))
		#self.img = self.img[eimg > 0]
		self.img = self.img.reshape(-1, 5)
		
		#indices = torch.randint(0, int(self.img.shape[0]), (self.img.shape[0] // CONFIG.sample_decay,))
		#self.img = self.img[indices]
		
		self.pos = self.img[:, :2].cuda()
		self.vals = self.img[:, 2:].cuda()
		
		print(self.img.shape)
	
	def DrawSamples(self, sample_count):
		#noise = torch.randn(self.pos.shape).cuda() * CONFIG.pnoise
		
		return self.pos, self.vals
		
		indices = torch.randint(0, int(self.img.shape[0]), (sample_count,))
		
		samples = self.img[indices]
		
		pos = samples[:, :2].cuda()
		vals = samples[:, 2:].cuda()
		
		return pos, vals
	
	def ImageGrid(self, height, width):
		return torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing = "ij"), dim = -1)