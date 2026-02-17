import time
import math
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import CONFIG
import DataLoader
import LinearModel

class Trainer:
	def __init__(self):
		pass

	def Train(self, epochs):
		losser = torch.nn.MSELoss()
		#losser = torch.nn.L1Loss()
		#losser = torch.nn.GaussianNLLLoss()
		
		loader = DataLoader.DataLoader()
		
		model = LinearModel.LinearModel()
		model = model.cuda()
		
		optimizer = optim.Adam(
			model.parameters(),
			lr = CONFIG.learning_rate,
		)
		
		print("\n\nTraining...")

		avg_time = 1
		start_total = time.time()

		for e in range(epochs + 1):
			start_time = time.time()

			pos, vals = loader.DrawSamples(CONFIG.batch_size)
			preds = model(pos)

			loss = losser(preds, vals)
			#loss = losser(preds, vals, 1 * torch.ones(preds.shape[0], 1, requires_grad = True).cuda())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			if e == 0:
				avg_time = (time.time() - start_time) / 7
			
			self.PrintUpdate(epochs, e, avg_time, loss, preds, loader)

			avg_time = (avg_time + (time.time() - start_time)) / 2
		
		print("\nCompleted in", int((time.time() - start_total) / 60), "minutes.")
		print("Final loss:", loss.item())
		
		model.Save("data/models/" + CONFIG.model_name)
		
		return model

	def PrintUpdate(self, epochs, e, avg_time, loss, preds, loader):

		remaining_epochs = epochs - e

		if self.ShouldPrint_E(epochs, e):
			eta = avg_time * remaining_epochs
			eta = eta / 60
			eta = "{:.2f}".format(eta)

			completion = str(100 *(e / (epochs)))
			completion = completion[:4] + "%"

			print("   [", e, "/", epochs, ":", completion,  "]")
			
			preds = preds.reshape(loader.img_grid.shape[0], loader.img_grid.shape[1], 3)
			preds = torch.clip(preds, 0, 1)
			preds = preds.detach().cpu().numpy()
			preds = (preds * 255).astype(np.uint8)
			cv2.imwrite("data/out/frames/" + str(e) + ".jpg", preds)
			
			print("    Loss	 :", loss.item())
			print("")
			print("    Batches left :", remaining_epochs)
			print("    Avg. Time    :", "{:.2f}".format(avg_time), "s")
			print("    ETA	      :", eta, "mins")
			print("\n")

	def ShouldPrint_E(self, epochs, e):
		if epochs < CONFIG.print_every_epoch:
			return True

		return e % CONFIG.print_every_epoch == 0 or e == epochs - 1
