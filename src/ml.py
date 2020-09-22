import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import random
from tensorflow import keras

# data.pkl:
# checkpoint_times = [x for x in range(2, 60, 2)] + [x for x in range(60, 120, 5)] + [120]
# df = pd.DataFrame(data=[[x for x in p.scores] + [x for x in p.comments] for p in done], 
# 		columns=['scores ' + str(x) for x in post.checkpoint_times] + ['comments ' + str(x) for x in post.checkpoint_times],
# 		index=[p.id for p in done])

# data-xx.pkl:
# checkpoint_times = [x for x in range(2, 62, 2)]
# df = pd.DataFrame(data=[[x for x in p.scores] + [x for x in p.comments] + [p.time] for p in done], 
# 	columns=['scores ' + str(x) for x in post.checkpoint_times] + ['comments ' + str(x) for x in post.checkpoint_times] + ['time'],
# 	index=[p.id for p in done])

df = pd.read_pickle('data.pkl')
df = df.astype('float32')

inputs = df.to_numpy()
n_data = inputs.shape[0]
print(n_data)
print(inputs.shape)
for i in range(n_data):
	for j in range(84):
		inputs[i][j] = min(max(0, inputs[i][j]), 1000) / 1000

import praw
reddit = praw.Reddit()
ending_scores = []
i = 0
for index in df.index:
	submission = reddit.submission(index)
	ending_scores.append(submission.score)
	print('got score', i)
	i += 1
print(ending_scores)

ending_scores = [min(max(x, 0), 99999) for x in ending_scores]

outputs = (np.log1p(ending_scores) - np.log(2)) / (np.log(100000) - np.log(2))

print(outputs)

n_train = n_data - n_data // 10

train_in = inputs[:n_train]
train_out = outputs[:n_train]
test_in = inputs[n_train:]
test_out = outputs[n_train:]

model = keras.Sequential([
    keras.layers.Dense(64, input_dim=inputs.shape[1]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(train_in, train_out, epochs=10)

test_loss = model.evaluate(test_in, test_out)

