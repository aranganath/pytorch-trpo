# This code demonstrates the constrained optimization problem
# We will be consider the rosenbrock function.
# All formulae for the Hessian and gradient information have 
# been computed exactly
import numpy as np
import torch


delta = 2
mu = 1e-2


#Numpy
nprosenbrock = lambda x: np.sum((1. - x[0])**2 + (x[1] - x[0]**2)**2)
npradconstraint =  lambda x: x[2]*(delta-(x[0]**2 + x[1]**2) - x[3])
npbarrier = lambda x: mu*np.log(x[3])
npfunc = lambda x: nprosenbrock(x) + npradconstraint(x) - nplogbarrier(x)

#Torch
torchrosenbrock = lambda x: torch.sum((1. - x[0])**2 + (x[1] - x[0]**2)**2) 
torchradconstraint =  lambda x: x[2]*(delta**2-(x[0]**2 + x[1]**2) - x[3])
torchbarrier = lambda x: mu*torch.log(x[3])
torchfunc = lambda x: torchrosenbrock(x) + torchradconstraint(x) - torchbarrier(x)

#Numpy version of the Hessian of the objective with respect to the parameters
def npN(x):
	grads = np.array([-2*(1-x[0]) + 2*(x[1]-x[0]**2) * (-2*x[0]) + x[2]*(-2*x[0]) , 2*(x[1] - x[0]**2) + x[2]*(-2*x[1])]).T
	grads = np.append(arr=grads,values=4 - x[0]**2 - x[1]**2 - x[3])
	grads = np.append(arr=grads, values=-x[2] - 10.0/x[3])
	N = np.array([[2 - 4*x[1] + 12*x[0]**2-2*x[2], -4*x[0], -2*x[0], 0],
		[-4*x[0], 2-2*x[2], -2*x[1], 0],
		[-2*x[0], -2*x[1], 0, -1],
		[0, 0, -1, 10.0/x[3]**2]])
	sol = - np.linalg.inv(N) @ grads
	return sol

def torchN(x):
	N = torch.autograd.functional.hessian(torchfunc, x)
	grads = torch.autograd.grad(torchfunc(x), x, retain_graph=True)[0]
	sol = - torch.linalg.inv(N) @ grads
	return sol


def numpylinesearch(x, sol, func):
	# Find the solution to the problem
	# Check if the linesearch sufficicently reduces the funciton
	# Take the step if it does
	merit = lambda x: nprosenbrock(x) - npbarrier(x)
	alpha = 0.9999
	max_iters = 20
	i=0
	while merit(x+sol) >= merit(x) and i < max_iters:
		sol*=alpha
		i+=1

	if i < max_iters:
		return x + sol
	else:
		return x

def torchlinesearch(x, sol, func):
	# Find the solution to the problem
	# Check if the linesearch sufficicently reduces the funciton
	# Take the step if it does
	# merit = lambda x: torchrosenbrock(x) - torchbarrier(x)
	merit = lambda x: torchfunc(x)
	alpha = 0.9999
	max_iters = 20
	i=0
	from pdb import set_trace
	set_trace()
	while merit(x+sol) >= merit(x) and i < max_iters:
		sol*=alpha
		i+=1

	if i < max_iters:
		return x + sol
	else:
		return x


# start = np.array([0.0,0.0,2.0,1.0])
start = torch.tensor([0.,0.,1e-2,1e2], requires_grad=True)
print(start.data)
alpha = 1.2
for i in range(10):
	sol = torchN(start)
	start = torchlinesearch(start, sol, npfunc)
	mu *= alpha
	print(start.data)
	print('mu:', mu)
	print('functional value:', torchfunc(start))

# Trash

