import numpy
import pandas
import sys
import torch
import torch.nn
import torch.nn.functional

class MoleculNN(torch.nn.Module):
    def __init__(self):
        super(MoleculNN, self).__init__()
        self.fc1 = torch.nn.Linear(1818, 20)

    def forward(self, x):
        x = self.fc1(x)
        return torch.nn.functional.softmax(x, dim=0)

def read_csv(filename):
    data = pandas.read_csv(filename).to_numpy()
    print("Data shape:", data.shape)
    X = data[:,8:].astype(float)
    T = data[:,5]
    print("Input shape: ", X.shape)
    print("Labels shape: ", T.shape)
    return X, T

if __name__ == "__main__":
    X, T = read_csv(sys.argv[1])
    print(X[0,:])
    x = torch.from_numpy(X[0])
    x = x.to(torch.float32)
    molNN = MoleculNN()
    y = molNN.forward(x)
    print(y)
    # forward(x)
