import numpy as np
import torch
from proj_code import TrainAndTarDataset, SimpleConvForecast

X = np.load('trainandtar/train_input.npy')
Y = np.load('trainandtar/train_target.npy')
print('X', X.shape, 'Y', Y.shape, 'nan in X', np.isnan(X).any(), 'nan in Y', np.isnan(Y).any())

# run one training step

dataset = TrainAndTarDataset(X, Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
model = SimpleConvForecast(in_channels=X.shape[1]*X.shape[4])
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.MSELoss()

x, targets = next(iter(loader))
print('batch x', x.shape, 'targets', {k: v.shape for k, v in targets.items()})

out = model(x)
print('out', out.shape, 'nan', torch.isnan(out).any().item(), 'min', out.min().item(), 'max', out.max().item())

y = targets['fire_probability']
loss = criterion(out, y)
print('loss', loss.item())

loss.backward()
print('grad nan', any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None))
