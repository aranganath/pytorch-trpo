# This code checks if the method performs correctly on
# some self defined toy problem with linear constraints

import torch
import numpy as np

# Define the full lagrangian, log-barrier setup
x = torch.tensor(np.array([1.0, 2.0]), requires_grad=True)
w = torch.tensor(6., requires_grad=True)
lamb = torch.tensor(2., requires_grad=True)
mu = torch.tensor(100.0, requires_grad=False)

def function(x, lamb, w, mu):
	return -x[0]*x[1] + lamb*(x[0]+ 2*x[1] - w - 4) + mu*torch.sum(torch.log(w))

def constraint(x, lamb, w):
	return lamb*(x[0]+ 2*x[1] - w - 4)

def optimization(function, constraint, x, w, lamb, mu):
	y = function(x, lamb, w, mu)
	c = constraint(x, lamb, w) 
	xgrad = torch.autograd.grad(y, x, retain_graph=True)[0]
	wgrad = torch.autograd.grad(y, w, retain_graph=True)[0]
	A = torch.autograd.grad(c, x, retain_graph= True)[0].unsqueeze(0).T
	H = torch.autograd.functional.hessian(function, inputs = (x, lamb, w, mu))[0][0]
	N = H + (w/lamb)* (A @ A.T)
	dx = - torch.inverse(N) @ xgrad.unsqueeze(1) + (mu/w)*(torch.inverse(N) @ A) + (lamb/w)*torch.inverse(N) @ A*(-c)
	from pdb import set_trace
	set_trace()
	dw = - A.T @ torch.inverse(N) @ xgrad.unsqueeze(1) + (mu/w)* A.T @ torch.inverse(N) @ A - (1 - (lamb/w)*(A.T @ torch.inverse(N) @ A))
	dlamb = 
	return dx, dw, dlamb
	

optimization(function, constraint, x, w, lamb, mu)




	# return dx, dw, dLambda
