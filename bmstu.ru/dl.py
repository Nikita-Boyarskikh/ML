#!/usr/bin/env python3
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

import numpy as np
import pandas as p

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def to_color(x):
    if x == 2:
        return 'red'
    elif x == 4:
        return 'blue'
    elif x == 5:
        return 'green'
    elif x == 3:
        return 'orange'
    else:
        return 'black'

dataset = p.DataFrame()
result = p.DataFrame()
for group in sys.argv[1:]:
    dataset = dataset.append(p.read_json('./data/' + group + '.json'))
    result = result.append(p.read_json('./results/' + group + '.json'))

result = list(map(lambda x: x[0], result.applymap(to_color).as_matrix().tolist()))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataset[0], dataset[1], dataset[2], c=result)
ax.set_xlabel('1 Module')
ax.set_ylabel('2 Module')
ax.set_zlabel('3 Module')
plt.show()

h = 0.02
dataset = StandardScaler().fit_transform(dataset)
dataset_train, dataset_test, result_train, result_test = train_test_split(dataset, result, test_size=.4, random_state=42)

x_min, x_max = dataset[:, 0].min() - .5, dataset[:, 0].max() + .5
y_min, y_max = dataset[:, 1].min() - .5, dataset[:, 1].max() + .5
z_min, z_max = dataset[:, 2].min() - .5, dataset[:, 2].max() + .5
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

clf = SVC(gamma=2, C=1)
clf.fit(dataset_train, list(result_train))
score = clf.score(dataset_test, result_test)
print('score: ', score)

print(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, zz, Z, cmap=cm, alpha=.8)

ax.scatter(dataset_train[:, 0], dataset_train[:, 1], dataset_train[:, 2], c=result_train, cmap=cm_bright, edgecolors='k')
ax.scatter(dataset_test[:, 0], dataset_test[:, 1], dataset_train[:, 2], c=result_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(zz.min(), zz.max())
ax.set_xticks(())
ax.set_yticks(())
ax.set_zticks(())

plt.tight_layout()
plt.show()
