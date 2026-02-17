import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import CONFIG
import DataLoader
import LinearModel
	
class Validater:
	def __init__(self):
		pass
	
	def Validate(self):
		
		loader = DataLoader.DataLoader()
		model = LinearModel.LinearModel().cuda()		
		model.Load("data/models/" + CONFIG.model_name)
		
		grid_list = loader.img_grid[:, :, :2].reshape(-1, 2).cuda().float() / torch.FloatTensor([loader.img_grid.shape[0], loader.img_grid.shape[1]]).cuda()
		
		canvas = model(grid_list)
		canvas = canvas.reshape(loader.img_grid.shape[0], loader.img_grid.shape[1], 3)
		canvas = torch.clip(canvas, 0, 1)
		canvas = canvas.detach().cpu().numpy()
		
		canvas = (canvas * 255).astype(np.uint8)
		
		cv2.imwrite("data/out/" + CONFIG.img_name, canvas)
		cv2.imshow("canvas", canvas)
		cv2.waitKey(0)