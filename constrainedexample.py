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
torchconstraint =  lambda x, s: (10. - torch.norm(x[:2]) - s)

# Define log barrier
torchbarrier = lambda s: torch.log(s)

#Define the optimization problem
torchfunc = lambda x, s, z: torchrosenbrock(x) - z*torchconstraint(x, s) - mu*torchbarrier(s)


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
	d2L_dt2 = torch.autograd.functional.hessian(torchfunc, (x,s,z))[0][0]

	AAT = 1/invSigma*A_I.outer(A_I)
	
	RedHess = d2L_dt2 + AAT
	return RedHess
	


def getGradient(f, c, s, x, z, A_I, invSigma):
	'''
	Computes the Hessian of torchfunc using the primal-dual system
	Inputs:
		f: objective function
		c: constraint  (\norm{x} < 100)
		
	
	Outputs:
		gradients

	'''

	# \nabla_x f
	df_dt = torch.autograd.grad(f(x), x)[0]
	RHS = -(df_dt - A_I*z) + 1/invSigma*A_I*(c(x,s) - mu/z)
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
	px = torch.linalg.inv(RedHess) @ RedGrad

	# Solve for 'z'
	pz = 1/invSigma*(torchconstraint(x,s) - mu/z - A_I.dot(px))
	
	# Solve for 's'
	ps = mu/z - s - invSigma*pz

	return px, ps, pz

def torchlinesearch(x, z, s, px, pz, ps, mu):
	'''
	Line search using the formula:
	\phi_v (x + \alpha_s p_x , s + \alpha_s p_s) \leq \phi(x_k, s_k) + \eta \alpha_s D \phi_v(x_k, s_k; p_w)

	Input: 
		x: input to torchrosenbrock function
		z: Lagrange multiplier to constraint
		s: radius
		px: Step on the primal variables
		ps: Step on the slack
		pz: Step on the lagrange multiplier

	Ouput:
		flag: updated or not
		x, z, s
	'''
	alphas = 1.
	flag = False
	
	# Define the merit function
	global nu
	torchphi = lambda x: torchrosenbrock(x) - mu*torch.log(s) + nu*torch.norm(torchconstraint(x, s))
	eta = 0.5
	iters = 0
	# print('--------------------Line-search----------------')
	xs = torch.cat([x, s])
	p = torch.cat([px, ps])
	while not flag and iters<num_iters:
		
		D = torch.autograd.grad(torchphi(xs), xs, create_graph=False)[0].dot(alphas*p)
		
		if torchphi(xs + alphas*p) <= torchphi(xs) + eta*D*alphas*torchphi(xs):
			with torch.no_grad():
				x += alphas*p[:2]
				s += alphas*p[2]
			flag = True
			continue
		# print('phi(x + alphax*px,s + alphas*ps):'+str(torchphi(alphas*px, alphas*ps)))
		# print('phi(x,s):'+str(torchphi(torch.zeros(px.shape), torch.zeros(ps.shape))))
		# print('-----------------------------------------------------------------')
		alphas*=0.5
		iters+=1

	if not flag and iters==num_iters:
		mu *= 2
		flag = True
		nu *= 0.5
		return (x,z,s, flag, mu)

	mu *= 0.5
	nu *= 2
	return (x,z,s, flag, mu)

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


	InvMat = tl.inv(A)
	InvMatdotv = InvMat.dot(v)
	
	# Now let's apply the Sherman-Morrison-Woodbury, which is given by the latex expression above
	InvMatOut = InvMat - InvMatdotV.outer(InvMatdotV)/(1+ v.dot(InvMatdotv))

	return InvMatOut



def prettyprint(delta, mu, func_val, iterate, iteration):

	'''
	Keeps stdout neat
	Input parameters:
		mu: coefficient of log-barrier
		iterate: current point
		func_val: Value of the rosenbrock function at the current point
		iteration: Iteration Number
	'''
	global nu
	print('Iteration:'+str(iteration+1)+'\t functional value:'+str(func_val.data.numpy()))
	print('Iterate:'+str(iterate.data.numpy())+'\t Iterate radius:' +str(torch.norm(iterate).data.numpy()))
	print('z:'+str(z.data.numpy())+' mu:'+str(mu)+'\t s:'+str(s[0].data.numpy())+'\t nu: '+str(nu)+'\n')



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--delta', type=float, default=2.0)
	parser.add_argument('--mu', type=float, default=1.0)
	parser.add_argument('--max-iters', type=int, default=30)
	parser.add_argument('--line-search-iters', type=int, default=10)
	parser.add_argument('--save', type=bool, default=False)
	parser.add_argument('--location', type=str, default='./rosenbrockresults/raw/path.pkl')
	args = parser.parse_args()
	delta = args.delta
	max_iters = args.max_iters
	mu = args.mu
	nu = 1.
	num_iters = 20
	iterates = []
	sol = torch.tensor([-1.,1.], requires_grad=True)
	s = torch.tensor([10.])
	z = torch.tensor([1e6])
	iterates.append(sol.clone().detach().data.numpy())	
	for i in range(max_iters):
		px, ps, pz = torchsolution(sol,z, s)
		func_val = torchrosenbrock(sol).data
		sol, z, s, flag, mu = torchlinesearch(sol, z, s, px, pz, ps, mu)
		iterates.append(sol.clone().detach().data.numpy())
		if flag:
			with torch.no_grad():
				sol += px.data
				s += ps.data
				z += pz.data

		prettyprint(delta, mu, func_val, sol, i)


	if args.save:
		with open(args.location, 'wb') as file:
			print('Results saved to ' + args.location)
			pkl.dump(iterates, file, protocol=pkl.HIGHEST_PROTOCOL)
