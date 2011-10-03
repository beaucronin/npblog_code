import numpy as np
import matplotlib.pyplot as plt
from metropolis import *

md = MetropolisDemo()

def plot_history(X, Y, Z, x, title=None, savefile=None, lines=True):
  fig = plt.figure()
  x2, x1 = zip(*x)
  plt.contour(X, Y, Z, 10)
  if lines:
    plt.scatter(x1, x2, c='black', marker='.', alpha=1., linestyle='solid', linewidth=3)
    plt.plot(x1, x2, 'k:', alpha=1.)
  else:
    plt.scatter(x1, x2, c='black', marker='.', alpha=.05, linestyle='solid', linewidth=3)
  plt.axis('equal')
  plt.xlim((-6,6))
  plt.ylim((-6,6))
  if title:
    plt.title(title)
  if savefile:
    fig.savefig(savefile)
  else:
    plt.show()

# Compute the target distribution on a mesh so we can plot the contours
x1 = np.arange(-4., 4., .1)
x2 = np.arange(-4., 4., .1)
X, Y = np.meshgrid(x1, x2)
Z = np.ndarray((len(x1), len(x2)))
for i in range(len(x1)):
	for j in range(len(x2)):
		Z[i,j] = prob((x1[i], x2[j]))

x = md.run(10)
plot_history(X, Y, Z, x, '10 iterations', 'out_1.png')
x = md.run(10)
plot_history(X, Y, Z, x, '20 iterations', 'out_2.png')
x = md.run(30)
plot_history(X, Y, Z, x, '50 iterations', 'out_3.png')
x = md.run(150)
plot_history(X, Y, Z, x, '200 iterations', 'out_4.png')
x = md.run(800)
plot_history(X, Y, Z, x, '1000 iterations', 'out_5.png')
x = md.run(4000)
plot_history(X, Y, Z, x, '5000 iterations', 'out_6.png', lines=False)
x = md.run(10000)
plot_history(X, Y, Z, x, '15000 iterations', 'out_7.png', lines=False)
x = md.run(25000)
plot_history(X, Y, Z, x, '40000 iterations', 'out_8.png', lines=False)

# for i in range(1000):
#   x = md.run(1)
#   plot_history(X, Y, Z, x, 'frames/out_%05d.png' % i)
# for i in range(1000,2000):
#   x = md.run(10)
#   plot_history(X, Y, Z, x, 'frames/out_%05d.png' % i)
# 
