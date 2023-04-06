# This code demonstrates the constrained optimization problem
# We will be consider the rosenbrock function.
# All formulae for the Hessian and gradient information have 
# been computed exactly
import numpy as np
import torch
import pickle as pkl
import argparse
from pdb import set_trace
from torch import linalg as tl


parser = argparse.ArgumentParser()


# Torch
# define the rosenbrock function
torchrosenbrock = lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# Define the constraint
torchconstraint =  lambda x,s: (x[0]**2 + x[1]**2 - s)

# Define log barrier
torchbarrier = lambda x: torch.log(s)

#Define the optimization problem
torchfunc = lambda x,s,z: torchrosenbrock(x) - z*torchconstraint(x, s) - mu*torchbarrier(x)

def getHessianPD(x,z,s, A_I, invSigma):
	'''
	Computes the Hessian of torchfunc using the primal-dual system
	Inputs:
		x: input to the rosenbrock fucntion (or whichever function is coded)
		z: Lagrange multiplier (coefficient of the constraint in the Primal)
		s: Radius of the constrained region
	
	Outputs:
		InvHessian: Inverse of the primal-dual system matrix

	'''
	d2L_dt2 = torch.autograd.functional.hessian(torchfunc, (x,z,s))[0][0]

	AAT = 1/invSigma*A_I.outer(A_I)
	
	RedHess = d2L_dt2 + AAT
	return RedHess
	


def getGradient(f, c, s, x, z, A_I, invSigma):
	'''
	Computes the Hessian of torchfunc using the primal-dual system
	Inputs:
		f: objective function
		c: constraint
		
	
	Outputs:
		gradients

	'''

	# \nabla_x f
	df_dt = torch.autograd.grad(f(x), x)[0]
	RHS = -(df_dt - A_I*z) + 1/invSigma*A_I *(c(x,s) - mu/z)
	return RHS

def torchsolution(x,z,s):
	'''
	Computes the step for the nonlinear interior point method
	Inputs:
		x: parameters of the function (main objective function)
		z: Lagrange multiplier (coefficient of the constraint in the lagrangian term)
		s: Bounds on the constraint (log barrier argument)
	'''

	# Constructing the Hessian matrix of the Primal-dual system
	A_I = torch.autograd.grad(torchconstraint(x,s), x)[0]
	invSigma = s/z
	RedHess = getHessianPD(x, z, s, A_I, invSigma)

	# Compute the corresponding gradient
	RedGrad = getGradient(torchrosenbrock, torchconstraint, s, x, z, A_I, invSigma)
	
	# Solve for 'x'
	set_trace()
	px = torch.linalg.inv(RedHess) @ RedGrad

	# Solve for 'z'
	pz = 1/invSigma*(torchconstraint(x,s) - mu/z - A_I.dot(px))
	
	# Solve for 's'
	ps = s - mu/z - invSigma*pz

	return px, ps, pz

def getSMW(A, v):

	'''
	Using the Sherman-Morrison-Woodbury formula to compute the inverse of the Hessian matrix
	
	Input parameters: 
		A: The hessian matrix
		v: The vector for computing the outer-product with

	Output:
		InvMatOut: $$(A + vv^T)^{-1} = \frac {A^{-1} + A^{-1}vv^{T}A^{-1}} {1 + v^{T}Av}$$

	'''
	
	# First let's compute the inverse of the matrix since it can be re-used multiple times.
	# Also the dot product between InvMat and v


	InvMat = tl.inv(matrix)
	InvMatdotv = InvMat.dot(v)
	
	# Now let's apply the Sherman-Morrison-Woodbury, which is given by the latex expression above
	InvMatOut = InvMat - InvMatdotV.outer(InvMatdotV)/(1+ v.dot(InvMatdotv))

	return InvMatOut

def prettyprint(delta, mu, func_val, start, iteration):
	print('Iteration:'+str(iteration)+'\t functional value:'+str(func_val.data.numpy()))
	print('Iterate:'+str(start.data.numpy())+'\t Iterate radius:' +str(torch.norm(start).data.numpy()))
	print('z:'+str(z.data)+' mu:'+str(mu)+'\t s:'+str(s[0].data.numpy())+'\n')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--delta', type=float, default=2.0)
	parser.add_argument('--mu', type=float, default=1.0)
	parser.add_argument('--max-iters', type=int, default=100)
	parser.add_argument('--line-search-iters', type=int, default=10)
	parser.add_argument('--save', type=bool, default=False)
	parser.add_argument('--location', type=str, default='./rosenbrockresults/raw/path.pkl')
	args = parser.parse_args()
	delta = args.delta
	max_iters = args.max_iters
	mu = args.mu
	iterates = []
	sol = torch.tensor([-1.,1.], requires_grad=True)
	s = torch.tensor([100.])
	z = torch.tensor([1e3])
	iterates.append(sol.data.numpy())
	for i in range(max_iters):
		px, ps, pz = torchsolution(sol,z, s)
		with torch.no_grad():
			sol += px.data
			s += ps.data
			z += pz.data
		iterates.append(sol.data.numpy())
		mu*=0.1
		func_val = torchrosenbrock(sol).data
		prettyprint(delta, mu, func_val, sol, i)

	if args.save:
		with open(location, 'wb') as file:
			print('Results saved to ' + args.save)
			pkl.dump(iterates, file, protocol=pkl.HIGHEST_PROTOCOL)
