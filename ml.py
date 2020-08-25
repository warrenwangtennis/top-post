import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import random
from tensorflow import keras

# timestamp = 1598385600
# dt = datetime.fromtimestamp(timestamp)

# print(dt)
# print(dt.hour)

df = pd.read_pickle('data.pkl')
df = df.astype('float32')


inputs = df.to_numpy()
n_data = inputs.shape[0]
print(n_data)
print(inputs.shape)
for i in range(n_data):
	for j in range(84):
		inputs[i][j] = min(max(0, inputs[i][j]), 1000) / 1000

# outputs = np.array([random.random() for i in range(n_data)])
outputs = np.array([.17 for i in range(n_data)])


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

