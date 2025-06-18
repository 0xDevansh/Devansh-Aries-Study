import torch
import mnist_loader
import numpy as np

lr = 0.00001
batch_size = 100
lamda = 1

train_data, validation_data, test_data = mnist_loader.load_data()
[x, y_t] = train_data
x = torch.from_numpy(x).float()[:1000] # taking only first 1000 to find good hyperparameters first
[x_v, y_v] = test_data
x_v = torch.from_numpy(x_v).float()

# need to transform y in the proper form
new_y = []
for num in y_t:
    vec = [1 if i == num else 0 for i in range(10)]
    new_y.append(vec)
y = torch.tensor(new_y).float()[:1000]

new_y = []
for num in y_v:
    vec = [1 if i == num else 0 for i in range(10)]
    new_y.append(vec)
y_v = torch.tensor(new_y).float()


model = torch.nn.Sequential(
    torch.nn.Linear(784, 30),
    torch.nn.Sigmoid(),
    torch.nn.Linear(30, 10),
    torch.nn.Softmax()
)
optimiser = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=lamda)

loss_fn = torch.nn.MSELoss(reduction='sum')

for e in range(50000):
    # split batch
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    losses = []
    for j in range(0, len(train_data), batch_size):
        x_batch = x[j:j+batch_size, :]
        y_batch = y[j:j+batch_size, :]

        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        losses.append(loss.item())

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    if e % 100 == 0:
        print(e, sum(losses))
    if e % 1000 == 0:
        # test on validation set
        valid_pred = model(x_v)
        valid_loss = loss_fn(valid_pred, y_v)
        print(f'Validation loss: {valid_loss.item()}')

class NNModel(torch.nn.Module):
    def __init__(self, lr=0.00001, batch_size=100, lamda=1):
        super(NNModel, self).__init__()
        self.linear1 = torch.nn.Linear(784, 30),
        self.activation1 = torch.nn.Sigmoid(),
        self.linear2 = torch.nn.Linear(30, 10),
        self.activation2 = torch.nn.Sigmoid(),
        self.optimiser = torch.optim.RMSprop(self.parameters, lr=self, weight_decay=lamda)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        return x
    