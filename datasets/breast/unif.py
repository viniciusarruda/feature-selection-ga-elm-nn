import numpy as np

filepath = 'wdbc.data'

with open(filepath) as f:
	data = f.readlines()

data = [d.strip().split(',') for d in data]
data = [['0' if i == 'M' else i for i in l] for l in data]
data = [['1' if i == 'B' else i for i in l] for l in data]

data = np.array(data, dtype=np.float)

Y = data[:, 1][:, np.newaxis]
X = data[:, 2:]

data = np.concatenate((X, Y), axis=1)

print(data.shape)

np.save('breastEW.npy', data)
