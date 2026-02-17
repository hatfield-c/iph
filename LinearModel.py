import torch
import numpy as np

import CONFIG

class LinearModel(torch.nn.Module):
	def __init__(self):
		super().__init__()

		nodes = CONFIG.nodes
		depth = CONFIG.depth

		self.linears = []
		self.norms = []
		
		for i in range(depth):
			insize = nodes
			outsize = nodes
			
			if i == 0:
				insize = 2
				
			linear = torch.nn.Linear(insize, outsize).cuda()
			norm = torch.nn.BatchNorm1d(nodes)
			
			self.linears.append(linear)
			self.norms.append(norm)
			
		self.linears = torch.nn.ModuleList(self.linears)
		self.norms = torch.nn.ModuleList(self.norms)
		self.depth = depth

		self.out_layer = torch.nn.Linear(nodes, 3).cuda()

		self.relu = torch.nn.ReLU()
		#self.act = torch.nn.ReLU()
		self.act = torch.nn.Hardtanh()
		#self.act = torch.nn.Hardshrink()
		#self.act = torch.nn.Softsign()
		#self.act = torch.nn.Tanhshrink()
		#self.act = torch.nn.Sigmoid()
		#self.act = torch.nn.Mish()
		#self.act = torch.nn.Softshrink()

	def forward(self, data):
		out = data
		
		for i in range(self.depth):
			linear = self.linears[i]
			norm = self.norms[i]
			
			out = linear(out)
			out = norm(out)
			
			out = torch.exp((-1) * torch.square(out - 1) / 2)
			#out = self.act(out)
		
		out = self.out_layer(out)
		
		return out
	
	def Save(self, save_path):
		torch.save(self.state_dict(), save_path)

		print("\nSaved model to", save_path)

	def Load(self, load_path):
		self.load_state_dict(torch.load(load_path))
		self.eval()

		print("\nLoaded model from", load_path)

		