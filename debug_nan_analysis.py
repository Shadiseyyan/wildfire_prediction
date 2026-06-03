import numpy as np

X = np.load('trainandtar/train_input.npy')
print('shape', X.shape)
print('nan count', np.isnan(X).sum())
idx = np.argwhere(np.isnan(X))
print('first nan idx', idx[0] if len(idx) else None)
if len(idx):
    i0, i1, i2, i3, i4 = idx[0]
    print('slice', X[i0, i1, i2, i3, :5])
