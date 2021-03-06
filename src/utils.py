from torch.utils.data import Dataset, DataLoader, random_split
import os
import pandas as pd
import torch
from datetime import datetime
import numpy as np


MIN_STARTING_SCORE = 0
MAX_STARTING_SCORE = 2000
MIN_ENDING_SCORE = 0
MAX_ENDING_SCORE = 200000
MIN_STARTING_COMMENTS = 0
MAX_STARTING_COMMENTS = 1000
MIN_ENDING_COMMENTS = 0
MAX_ENDING_COMMENTS = 100000

def normalize_log_with_bounds(x, min_x, max_x):
	x = max(min(x, max_x), min_x) - min_x
	x = np.log1p(x) / np.log1p(max_x - min_x)
	assert 0 <= x <= 1
	return x

def normalize_starting_score(score):
	return normalize_log_with_bounds(score, MIN_STARTING_SCORE, MAX_STARTING_SCORE)

def normalize_ending_score(score):
	return normalize_log_with_bounds(score, MIN_ENDING_SCORE, MAX_ENDING_SCORE)

def normalize_starting_comments(comments):
	return normalize_log_with_bounds(comments, MIN_STARTING_COMMENTS, MAX_STARTING_COMMENTS)

def normalize_ending_comments(comments):
	return normalize_log_with_bounds(comments, MIN_ENDING_COMMENTS, MAX_ENDING_COMMENTS)

def normalize_timestamp(timestamp):
	dt = datetime.fromtimestamp(timestamp)
	return ((dt.minute + dt.second / 60) / 60 + dt.hour) / 24

class PostDataset(Dataset):

	def __init__(self, dataset_path):
		input_df = pd.read_pickle(dataset_path + '/data-951.pkl')
		assert input_df.shape == (982, 61)
		output_df = pd.read_pickle(dataset_path + '/output_data.pkl')
		assert output_df.shape == (982, 2)
		assert(input_df.index.equals(output_df.index))

		# print(output_df)
		# exit()


		input_data = torch.empty(input_df.shape)
		output_data = torch.empty(output_df.shape)
		for i in range(input_df.shape[0]):
			for j in range(30):
				input_data[i][j] = normalize_starting_score(input_df.iloc[i, j])
			for j in range(30, 60):
				input_data[i][j] = normalize_starting_comments(input_df.iloc[i, j])
			input_data[i][60] = normalize_timestamp(input_df.iloc[i, 60])

			output_data[i][0] = normalize_ending_score(output_df.iloc[i][0])
			output_data[i][1] = normalize_ending_comments(output_df.iloc[i][1])

		self.data = [*zip(input_data, output_data)]		

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		"""
		@return (Tensor, Tensor)
		"""
		if not(0 <= idx < len(self.data)):
			raise IndexError()
		return self.data[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
	dataset = PostDataset(dataset_path)
	return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)

def load_data_split(dataset_path, num_workers=0, batch_size=128, train_ratio=.9):
	dataset = PostDataset(dataset_path)
	train_len = int(len(dataset) * train_ratio)
	data_split = random_split(dataset, [train_len, len(dataset) - train_len])
	return (DataLoader(data_split[0], num_workers=num_workers, batch_size=batch_size, shuffle=True),
			DataLoader(data_split[1], num_workers=num_workers, batch_size=batch_size, shuffle=True)
		)

if __name__ == '__main__':
	# dataset = PostDataset('big_data_start_9-16-2020-1630-1930')
	data_train, data_test = load_data_split('big_data_start_9-16-2020-1630-1930', num_workers=4, batch_size=16)
	for x, y in data_train:
		pass
	for x, y in data_test:
		pass