import numpy as np
print np.load('data/class_mean.npy')*10
a1 = np.load('data/a8.npy')
a2 = np.load('data/three_value_a8.npy')
# a1:thin thick light dense flexible stiff streachable notstreachable
# a2:stiff flexible streachable notstreachable thick thin dense notdense
print a1
# 54761023
a1 = a1[:, [5,4,6,7,1,0,3,2]]
print a1
print a2
import sys
print sys.path


