# This code demonstrates the constrained optimization problem
# We will be consider the rosenbrock function.
# All formulae for the Hessian and gradient information have 
# been computed exactly
import numpy as np
import torch
import pickle as pkl
import argparse
from pdb import set_trace

parser = argparse.ArgumentParser()


# Torch
# define the rosenbrock function
torchrosenbrock = lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# Define the constraint
torchradconstraint =  lambda x: (x[0]**2 + x[1]**2 - x[2])

# Define log barrier
torchbarrier = lambda x: torch.log(x[2])

#Define the optimization problem
torchfunc = lambda x: torchrosenbrock(x) - x[3]*torchradconstraint(x) - mu*torchbarrier(x)

def torchsolution(x):
	N = torch.autograd.functional.hessian(torchfunc, x)
	grads = torch.autograd.grad(torchfunc(x), x, retain_graph=True)[0]
	sol = - torch.linalg.inv(N) @ grads
	return sol

def prettyprint(delta, mu, func_val, start, iteration, flag):
	print(start.data)
	
	print('Iteration:'+str(iteration)+'\t functional value:'+str(func_val)+'\n')
	print('mu:'+str(mu)+'\t Used line-search:'+str(flag))



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--delta', type=float, default=2.0)
	parser.add_argument('--mu', type=float, default=10.0)
	parser.add_argument('--max-iters', type=int, default=100)
	parser.add_argument('--line-search-iters', type=int, default=10)
	parser.add_argument('--save', type=bool, default=False)
	parser.add_argument('--location', type=str, default='./rosenbrockresults/raw/path.pkl')
	args = parser.parse_args()
	delta = args.delta
	max_iters = args.max_iters
	mu = args.mu
	iterates = []
	sol = torch.tensor([10.,10., 2., 1e6], requires_grad=True)
	print(sol.data.numpy())
	iterates.append(sol.data.numpy())
	for i in range(max_iters):
		delsol = torchsolution(sol).data
		with torch.no_grad():
			sol = sol + delsol.data
		iterates.append(sol.data.numpy())
		func_val = torchrosenbrock(sol).data
		flag = False
		mu = 0.01*mu
		sol.requires_grad=True
		prettyprint(delta, mu, func_val, sol, i, flag)

	if args.save:
		with open(location, 'wb') as file:
			print('Results saved to ' + args.save)
			pkl.dump(iterates, file, protocol=pkl.HIGHEST_PROTOCOL)
