from .models import L2Loss, model_factory, save_model
from .utils import accuracy, load_data, load_data_split
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_train, data_test = load_data_split('big_data_start_9-16-2020-1630-1930', num_workers=4, batch_size=16)

def train(args, lr=1e-3, epochs=25):
	model = model_factory[args.model]().to(device)

	loss_func = L2Loss()
	optim = torch.optim.Adam(model.parameters(), lr=lr)

	model.train()
	for epoch in range(epochs):
		for x, y in data_train:
			x = x.to(device)
			y = y.to(device)
			y_pred = model(x)

			loss = loss_func(y_pred, y)
			loss.backward()
			optim.step()
			optim.zero_grad()

	model.eval()
	losses_test = []
	for x, y in data_test:
		x = x.to(device)
		y = y.to(device)

		y_pred = model(x)
		losses_test.append(loss_func(y_pred, y))

	loss_test_avg = torch.FloatTensor(losses_test).mean().item()

	save_model(model)

	return loss_test_avg


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model', choices=['mlp'], default='mlp')

	args = parser.parse_args()
	train(args)
