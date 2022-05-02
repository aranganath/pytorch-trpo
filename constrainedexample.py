# This code checks if the method performs correctly on
# some self defined toy problem with linear constraints

import torch
import numpy as np

# Define the full lagrangian, log-barrier setup
x = torch.tensor(np.array([1.0, 2.0]), requires_grad=True)
w = torch.tensor(10., requires_grad=True)
lamb = torch.tensor(2., requires_grad=True)
mu = torch.tensor(1.0, requires_grad=False)
beta = 10

function = lambda x, lamb, w, mu: -x[0]*x[1] + lamb*(x[0]+ 2*x[1] - w - 4) + mu*torch.log(w)
constraint = lambda x,lamb, w: lamb*(x[0]+ 2*x[1] - w - 4)

rho = lambda x,lamb, w: w - (lamb*(x[0]+ 2*x[1] - w - 4))
psi =  lambda x,w,lamb,mu: function(x, lamb, w, mu) - mu*torch.log(w) + beta/2 * torch.norm(constraint(x,lamb,w))**2

def optimization(x, w, lamb, mu):
	y = function(x, lamb, w, mu)
	c = constraint(x, lamb, w) 
	xgrad = torch.autograd.grad(y, x, retain_graph=True)[0]
	wgrad = torch.autograd.grad(y, w, retain_graph=True)[0]
	A = torch.autograd.grad(c, x, retain_graph= True)[0]
	H = torch.autograd.functional.hessian(function, inputs = (x, lamb, w, mu))[0][0]
	N = H + (w/lamb)* (A.unsqueeze(1) @ A.unsqueeze(1).T)
	sigma = xgrad.unsqueeze(1) - lamb* A.unsqueeze(1)
	gamma = mu/w - lamb
	dx = torch.inverse(N) @ (A.unsqueeze(1)*((lamb/w)*constraint(x,lamb,w) + gamma)-sigma)
	dlamb = - (mu/w**2)*(x[0]+ 2*x[1] - w - 4 - A.unsqueeze(1).T @ dx - w**2/mu*lamb + w)
	dw = (w**2/lamb)*(lamb - dlamb)
	return dx, dw, dlamb


#Define the merit function here. We need to perform a line serach on this function
# We use the first order norm on this function. for the constraint
# Parameters:
# function: requires the function that we are minimizing
# constraint: requires the square of the 2-norm of the 
def linesearch(x, w, lamb, dx, dw, dlamb):
	f1 = psi(x, w, lamb, mu)
	f2 = psi(x+dx.view_as(x), w+dw.view_as(w), lamb+dlamb.view_as(lamb), mu)
	alpha=1
	iters = 0
	maxiters = 10
	while f2>f1 and iters<maxiters:
		from pdb import set_trace
		set_trace()
		alpha*=0.9
		f1 = psi(x, w, lamb, mu)
		f2 = psi(x+alpha*dx.view_as(x), w+alpha*dw.view_as(w), lamb+alpha*dlamb.view_as(lamb), mu)

	return dx, dw, dlamb

for i in range(5):
	dx, dw, dlamb = optimization(x, w, lamb, mu)
	dx, dw, dlamb = linesearch(x, w, lamb, dx, dw, dlamb)
	x += dx
	w += dw
	lamb+=dlamb
