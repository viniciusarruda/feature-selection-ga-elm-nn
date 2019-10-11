import numpy as np

filepath = 'hepatitis.data'

with open(filepath) as f:
	data = f.readlines()

data = [d.strip().split(',') for d in data]
data = [['0' if i == '?' else i for i in l] for l in data] # missed data replaced to zero

data = np.array(data, dtype=np.float)

# 1. Class: DIE, LIVE -- 1 and 2 --> set to 0 and 1
Y = data[:, 0] - 1.0

X = data[:, 1:]

# 3. SEX: male, female -- 1 and 2 --> set to -1 and 1
# 4. STEROID: no, yes
# 5. ANTIVIRALS: no, yes
# 6. FATIGUE: no, yes
# 7. MALAISE: no, yes
# 8. ANOREXIA: no, yes
# 9. LIVER BIG: no, yes
# 10. LIVER FIRM: no, yes
# 11. SPLEEN PALPABLE: no, yes
# 12. SPIDERS: no, yes
# 13. ASCITES: no, yes
# 14. VARICES: no, yes
# 20. HISTOLOGY: no, yes
for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20]:
	ii = i - 2
	X[X[:, ii] == 1, ii] = -1.0
	X[X[:, ii] == 2, ii] = 1.0

# 2. AGE: 10, 20, 30, 40, 50, 60, 70, 80
# 15. BILIRUBIN: 0.39, 0.80, 1.20, 2.00, 3.00, 4.00
# 16. ALK PHOSPHATE: 33, 80, 120, 160, 200, 250
# 17. SGOT: 13, 100, 200, 300, 400, 500, 
# 18. ALBUMIN: 2.1, 3.0, 3.8, 4.5, 5.0, 6.0
# 19. PROTIME: 10, 20, 30, 40, 50, 60, 70, 80, 90

data = np.concatenate((X, Y[:, np.newaxis]), axis=1)

np.save('hepatitis.npy', data)
