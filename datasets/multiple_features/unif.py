import numpy as np

filepath = 'mfeat-fac'
datas = []

for filepath in ['mfeat-fac', 'mfeat-fou', 'mfeat-kar', 'mfeat-mor', 'mfeat-pix', 'mfeat-zer']:

	with open(filepath) as f:
		data = f.readlines()
	data = [d.strip().split() for d in data]

	data = np.array(data, dtype=np.float)

	datas.append(data)

Y = np.zeros((2000, 1))
Y[:200, 0] = 0.0
Y[200:400, 0] = 1.0
Y[400:600, 0] = 2.0
Y[600:800, 0] = 3.0
Y[800:1000, 0] = 4.0
Y[1000:1200, 0] = 5.0
Y[1200:1400, 0] = 6.0
Y[1400:1600, 0] = 7.0
Y[1600:1800, 0] = 8.0
Y[1800:2000, 0] = 9.0

datas.append(Y)
data = np.hstack(datas)

np.save('multiple_features.npy', data)
