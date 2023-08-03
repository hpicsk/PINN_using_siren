"""
@author: Computational Domain
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from siren import *

nu = 0.01

class NavierStokes():
    def __init__(self, X, Y, T, u, v, omega=0.5):

        self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        self.y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(T, dtype=torch.float32, requires_grad=True)

        self.u = torch.tensor(u, dtype=torch.float32)
        self.v = torch.tensor(v, dtype=torch.float32)

        #null vector to test against f and g:
        self.null = torch.zeros((self.x.shape[0], 1))

        # initialize network:
        self.network()

        self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=1, max_iter=200000, max_eval=50000,
                                           history_size=50, tolerance_grad=1e-05, tolerance_change=0.5 * np.finfo(float).eps,
                                           line_search_fn="strong_wolfe")

        self.mse = nn.MSELoss()

        #loss
        self.ls = 0

        #iteration number
        self.iter = 0

    def network(self):

        self.net=Siren(in_features=3, out_features=2, hidden_features=20, 
                  hidden_layers=9, outermost_linear=True,first_omega_0=omega, hidden_omega_0=omega)
        # self.net = nn.Sequential(
        #     nn.Linear(3, 20), nn.Tanh(),
        #     nn.Linear(20, 20), nn.Tanh(),
        #     nn.Linear(20, 20), nn.Tanh(),
        #     nn.Linear(20, 20), nn.Tanh(),
        #     nn.Linear(20, 20), nn.Tanh(),
        #     nn.Linear(20, 20), nn.Tanh(),
        #     nn.Linear(20, 20), nn.Tanh(),
        #     nn.Linear(20, 20), nn.Tanh(),
        #     nn.Linear(20, 20), nn.Tanh(),
        #     nn.Linear(20, 2))

    def function(self, x, y, t):

        out = self.net(torch.hstack((x, y, t)))
        coords = out[1]
        res=out[0] 
        psi=res[:,0].unsqueeze(-1)
        p=res[:,1].unsqueeze(-1)
        # import pdb
        # pdb.set_trace()

        psi_coords_y=torch.autograd.grad(psi, coords, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        u=psi_coords_y[:,1].unsqueeze(-1)

        u_coords=torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x=u_coords[:,0].unsqueeze(-1)
        u_y=u_coords[:,1].unsqueeze(-1)
        u_t=u_coords[:,2].unsqueeze(-1)

        u_xx=torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0].unsqueeze(-1)
        u_yy=torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,1].unsqueeze(-1)

        psi_coords_x=-1.*torch.autograd.grad(psi, coords, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v=psi_coords_x[:,0].unsqueeze(-1)

        v_coords=torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_x=v_coords[:,0].unsqueeze(-1)
        v_y=v_coords[:,1].unsqueeze(-1)
        v_t=v_coords[:,2].unsqueeze(-1)

        v_xx=torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0].unsqueeze(-1)
        v_yy=torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,1].unsqueeze(-1)

        p_coords=torch.autograd.grad(p, coords, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_x = p_coords[:,0].unsqueeze(-1)
        p_y = p_coords[:,1].unsqueeze(-1)
            
        f = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
        g = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

        return u, v, p, f, g

    def closure(self):
        # reset gradients to zero:
        self.optimizer.zero_grad()

        # u, v, p, g and f predictions:
        u_prediction, v_prediction, p_prediction, f_prediction, g_prediction = self.function(self.x, self.y, self.t)

        # calculate losses
        u_loss = self.mse(u_prediction, self.u)
        v_loss = self.mse(v_prediction, self.v)
        f_loss = self.mse(f_prediction, self.null)
        g_loss = self.mse(g_prediction, self.null)
        self.ls = u_loss + v_loss + f_loss +g_loss

        # derivative with respect to net's weights:
        self.ls.backward()

        self.iter += 1
        if not self.iter % 1:
            print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.ls))

        return self.ls

    def train(self):

        # training loop
        self.net.train()
        self.optimizer.step(self.closure)

N_train = 5000

data = scipy.io.loadmat('cylinder_wake.mat')

## main loop ## 


# data normalization
# data['X_star'][:,0] = (data['X_star'][:,0]-4.5/3.5)
# data['X_star'][:,1] = data['X_star'][:,1]/2
# data['t'] = (data['t']-10)/10
# data['U_star']
# data['p_star']

U_star = data['U_star']  # N x 2 x T
P_star = data['p_star']  # N x T
t_star = data['t']  # T x 1
X_star = data['X_star']  # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

x_test = X_star[:, 0:1]
y_test = X_star[:, 1:2]
p_test = P_star[:, 0:1]
u_test = U_star[:, 0:1, 0]
t_test = np.ones((x_test.shape[0], x_test.shape[1]))

# Rearrange Data
XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T

UU = U_star[:, 0, :]  # N x T
VV = U_star[:, 1, :]  # N x T
PP = P_star  # N x T

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

u = UU.flatten()[:, None]  # NT x 1
v = VV.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1

# Training Data
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]

for omega in [3]:
    '''
    pinn = NavierStokes(x_train, y_train, t_train, u_train, v_train, omega)

    pinn.train()

    torch.save(pinn.net.state_dict(), f'siren_no_norm_omega{omega}.pt')
    '''

    pinn = NavierStokes(x_train, y_train, t_train, u_train, v_train)
    pinn.net.load_state_dict(torch.load(f'siren_no_norm_omega{omega}.pt'))
    pinn.net.eval()

    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
    y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True)
    t_test = torch.tensor(t_test, dtype=torch.float32, requires_grad=True)

    u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, t_test)

    with torch.no_grad():
        print("u mse loss: ", torch.nn.functional.mse_loss(torch.tensor(u_test),u_out))
        print("p mse loss: ", torch.nn.functional.mse_loss(torch.tensor(p_test),p_out))

"""
    u_plot = p_out.data.cpu().numpy()
    # import pdb
    # pdb.set_trace()
    u_plot = np.reshape(u_plot, (50, 100))

    fig, ax = plt.subplots()

    plt.contourf(u_plot, levels=30, cmap='jet')
    plt.colorbar()
    #plt.show()

    def animate(i):
        ax.clear()
        u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, i*t_test)
        u_plot = p_out.data.cpu().numpy()

        u_plot = np.reshape(u_plot, (50, 100))
        cax = ax.contourf(u_plot, levels=20, cmap='jet')
        plt.xlabel(r'$x$')
        plt.xlabel(r'$y$')
        plt.title(r'$p(x,\; y, \; t)$')

    # Call animate method
    ani = animation.FuncAnimation(fig, animate, 20, interval=1, blit=False)
    ani.save(f'siren_no_norm_omega{omega}_p_field_lbfgs.gif')
"""