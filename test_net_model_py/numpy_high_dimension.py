import numpy as np

b = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
              [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
              [[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]],
              ])

print(b.shape)
# print("b[0, ::],b[1, ::],b[-1, ::],b[0:2, ::]")
# print(b[0, ::], b[0, ::].shape)
# print(b[1, ::], b[1, ::].shape)
# print(b[-1, ::], b[-1, ::].shape)
# print(b[0:2, ::], b[0:2, ::].shape)
# print("b[:, 0:],b[:, 1:],b[:, -1:],b[:, 0:2:]")
print(b[:, 0:], b[:, 0:].shape)
# print(b[:, 1:], b[:, 1:].shape)
# print(b[:, -1:], b[:, -1:].shape)
# print(b[:, 0:2:], b[:, 0:2:].shape)
print("b[::, 0],b[::, 1],b[::, -1],b[::, 0:2:]")
print(b[::, 0], b[::, 0].shape)
print(b[::, 1], b[::, 1].shape)
print(b[::, -1], b[::, -1].shape)
print(b[::, 0:2:], b[::, 0:2].shape)
print(b[::,0, 0], b[::, 0, 0].shape)

print("b[:,:, 0],b[:,:, 1],b[:,:, -1],b[:,:, 0:2:]")
print(b[:, :, 0], b[:, :, 0].shape)
print(b[:, :, 1], b[:, :, 1].shape)
print(b[:, :, -1], b[:, :, -1].shape)
print(b[:, :, 0:2:], b[:, :, 0:2].shape)