import torch
from os import path

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
	def __init__(self, input_dim, output_dim, hidden_dims=[]):
		super().__init__()

		c = input_dim
		layers = []
		for dim in hidden_dims:
			layers.append(torch.nn.Linear(c, dim))
			layers.append(torch.nn.ReLU())
			c = dim
		layers.append(torch.nn.Linear(c, output_dim))

		self.seq = torch.nn.Sequential(*layers)

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
	for n, m in model_factory.items():
		if isinstance(model, m):
			return torch.save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
	raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
	r = model_factory[model]()
	r.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
	return r
