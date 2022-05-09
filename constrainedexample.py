# This code demonstrates the constrained optimization problem
# We will be consider the rosenbrock function.
# All formulae for the Hessian and gradient information have 
# been computed exactly
import numpy as np
import torch


lamb = 2
#Define the function
torchrosenbrock = lambda x: torch.sum((1. - x[0])**2 + (x[1] - x[0]**2)**2) 
torchradconstraint =  lambda x: x[2]*(0.25-(x[0]**2 + x[1]**2) - x[3])
torchbarrier = lambda x: 10.0*torch.log(x[3])
torchfunc = lambda x: torchrosenbrock(x) + torchradconstraint(x) - torchbarrier(x)

nprosenbrock = lambda x: np.sum((1. - x[0])**2 + (x[1] - x[0]**2)**2) 
npradconstraint =  lambda x: x[2]*(4-(x[0]**2 + x[1]**2) - x[3])
npbarrier = lambda x: 1e-10*np.log(np.abs(x[3]))
npfunc = lambda x: nprosenbrock(x) + npradconstraint(x) - nplogbarrier(x)


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


# x[2]: lamb
# x[3]: w
def torchN(x):
	N = torch.autograd.functional.hessian(torchfunc, x)
	grads = torch.autograd.grad(torchfunc(x), x, retain_graph=True)[0]
	return N, grads


def linesearch(x, sol, func):
	# Find the solution to the problem
	# Check if the linesearch sufficicently reduces the funciton
	# Take the step if it does
	merit = lambda x: nprosenbrock(x) - npbarrier(x)
	alpha = 0.9
	max_iters = 10
	i=0
	while merit(x+sol) >= merit(x) and i < max_iters:
		sol*=alpha
		i+=1

	if i < max_iters:
		return x + sol
	else:
		return x


# torchN, torchgrads = torchN(torch.tensor([10.0,10.0, 2.0,4.0], requires_grad=True))
# npN, npgrads = defineN()

# torchsol = -torch.inverse(torchN) @ torchgrads
# npsol = -np.linalg.inv(npN) @ npgrads

start = np.array([100.0,100.0,2.0,1.269579])

for i in range(20):
	sol = npN(start)
	print(start)
	start = linesearch(start, sol, npfunc)