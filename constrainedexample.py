# This code demonstrates the constrained optimization problem
# We will be consider the rosenbrock function.
# All formulae for the Hessian and gradient information have 
# been computed exactly
import numpy as np
import torch
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()

delta = 100
mu = 1e-2

# Torch
# define the rosenbrock function
torchrosenbrock = lambda x: torch.sum((torch.ones(x[0].shape) - x[0])**2 + (x[1] - x[0]**2)**2)

# Define the constraint
torchradconstraint =  lambda x: x[2]*(delta**2-(x[0]**2 + x[1]**2) - x[3])

# Define the log-barrier function
torchbarrier = lambda x: mu*torch.log(x[3])

#Define the optimization problem
torchfunc = lambda x: torchrosenbrock(x) + torchradconstraint(x) - torchbarrier(x)

def torchsolution(x):
	N = torch.autograd.functional.hessian(torchfunc, x)
	grads = torch.autograd.grad(torchfunc(x), x, retain_graph=True)[0]
	sol = - torch.linalg.inv(N) @ grads
	return sol


def torchlinesearch(x, sol, func):
	# Find the solution to the problem
	# Check if the linesearch sufficiently reduces the function
	# Take the step if it does
	# merit = lambda x: torchrosenbrock(x) - torchbarrier(x)
	merit = lambda x: torchfunc(x)
	max_iters = 10
	i=0
	alpha = 1
	grad = torch.autograd.grad(torchfunc(x), inputs=x)[0]
	gradsol = torch.autograd.grad(torchfunc(x+alpha*sol), inputs=x)[0]
	c1 = 0.5
	c2 = 0.5
	# from pdb import set_trace
	# set_trace()
	while i < max_iters and torchfunc(x+ alpha*sol)<= torchfunc(x) + c1*alpha*grad.dot(sol) and grad.dot(sol)>=c2*grad.dot(sol):
		alpha = 0.9*alpha
		i+=1

	if i < max_iters:
		print('Used line search')
		return x + alpha*sol
	else:
		return x
		


iterates = []
start = torch.tensor([3000.,3000.,1e-2,1], requires_grad=True)
print(start.data.numpy())
iterates.append(start.data.numpy())
for i in range(5):
	sol = torchsolution(start)
	start = torchlinesearch(start, sol, torchfunc)
	iterates.append(start.data.numpy())
	print(start.data)
	print('mu:', mu)
	print('functional value:', torchfunc(start).data)

with open('./rosenbrockresults/raw/path.pkl', 'wb') as file:
	print('Results saved to ./rosenbrockresults/raw/path.pkl')
	pkl.dump(iterates, file, protocol=pkl.HIGHEST_PROTOCOL)


