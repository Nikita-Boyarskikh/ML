import numpy as np

img = np.loadtxt('cars.txt', dtype=int)
img = img.astype(np.uint8)

Gx = np.array([[-1, 0, +1],
	       [-2, 0, +2],
	       [-1, 0, +1]])

Gy = np.array([[-1, -2, -1],
	       [0, 0, 0],
	       [+1, +2, +1]])

G = np.sqrt(Gx*Gx+Gy*Gy)
