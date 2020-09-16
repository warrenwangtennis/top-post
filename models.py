import torch
import torch.nn.functional as F


class L2Loss(torch.nn.Module):
	def forward(self, output, target):
		"""

		L2 Loss

		@output:  torch.Tensor((B,C))
		@target: torch.Tensor((B,C))

		@return:  torch.Tensor((,))
		"""
		return torch.mean((output - target) ** 2)


class MLPModel(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()

		self.seq = torch.nn.Sequential(
			torch.nn.Linear(input_dim, 64), 
			torch.nn.ReLU(), 
			torch.nn.Linear(64, 64),
			torch.nn.ReLU(), 
			torch.nn.Linear(64, output_dim))

	def forward(self, x):
		"""
		@x: torch.Tensor((B,3,64,64))
		@return: torch.Tensor((B,6))
		"""
		return self.seq(x.view(x.shape[0], -1))


model_factory = {
	'mlp': MLPModel,
}


def save_model(model):
	from torch import save
	from os import path
	for n, m in model_factory.items():
		if isinstance(model, m):
			return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
	raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
	from torch import load
	from os import path
	r = model_factory[model]()
	r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
	return r
