import numpy as np

def Dy(u):
    rows, cols = u.shape
    d = np.zeros_like(u)
    d[1:, :] = u[1:, :] - u[0:-1, :]
    d[0, :] = u[0, :] - u[-1, :]

    return d

def Dyt(u):
    rows, cols = u.shape
    d = np.zeros_like(u)
    d[0:-1, :] = u[0:-1, :] - u[1:, :]
    d[-1, :] = u[-1, :] - u[0, :]

    return d

def Dx(u):
    rows, cols = u.shape
    d = np.zeros_like(u)
    d[:, 1:] = u[:, 1:]-u[:, 0:-1]
    d[:, 0] = u[:, 0] - u[:, -1]

    return d

def Dxt(u):
    rows, cols = u.shape
    d = np.zeros_like(u)
    d[:, 0:-1] = u[:, 0:-1]-u[:, 1:]
    d[:, -1] = u[:, -1] - u[:, 0]

    return d


# function d = Dyt(u)
# [rows,cols] = size(u);
# d = zeros(rows,cols);
# d(1:rows-1,:) = u(1:rows-1,:)-u(2:rows,:);
# d(rows,:) = u(rows,:)-u(1,:);
# return