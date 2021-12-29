import numpy as np

a = np.arange(9).reshape((1, 3, 3))
a = a.astype(int)
print(a)
b = a[a>4]
b = b.astype(int)
print(b.shape)
c = np.arange(4).reshape((1,4))
c = c.astype(int)
print(c)
c = c.transpose(1,0)
print(c.shape)
d = np.zeros((1,3,3),dtype=np.uint8)
d = d.astype(int)
#print(d[a>4])
#d[a>4] = c
d = d.transpose(1,2,0)
e = d[ a>4 ]
print(e.shape)
#d[a>4] = c
d = d.transpose(2,0,1)
print(d)