import numpy as np

import torch
from torch.autograd import Variable
from utils import *
import scipy.linalg as sl
from pdb import set_trace
from sys import stdout

class InteriorPointMethod(object):
	def __init__(self, maxiters=100, maxhist=100, verbose=False):
		self.S = None
		self.Y = None
		self.SS = None
		self.YY = None

		self.SY = None
		self.prev_flat_grad = []
		self.flat_grad = []
		self.lr = 1e-5
		
		self.maxiters = maxiters
		self.maxhist = maxhist
		self.first = True
		self.mu = 0.9
		self.gamma = 1

		self.rhok = None
		self.delta = 10
		self.verbose = verbose
		self.eps = 1e-8
		self.iters = 0
		self.prev_theta = None
		self.z = Variable(torch.rand(1))
		self.s = Variable(torch.rand(1))


	def flatten(self, value):
		return torch.cat([val.view(-1) for val in value])

	def gather_flat_grad(self, model):
		views = []
		for p in model.parameters():
			p.retain_grad()
			if p.grad is None:
				view = p.new(p.numel()).zero_()
			elif p.grad.is_sparse:
				view = p.grad.to_dense().view(-1)
			else:
				view = p.grad.view(-1)
			views.append(view)
		return torch.cat(views, axis=0)


	def arclsr1(self, model, get_loss, get_kl, max_kl, damping, environment):
		# We use the gradient from the traditional loss function
		# But we use the hessian approximation using the kl-divergence
		# How do we do it ?
		# For the kl divergence, we will use the hessian
		self.mu = 0.9
		for it in range(self.maxiters):
			if self.prev_theta is not None:
				kl_loss = get_kl(self.prev_theta, model)
				kl_loss = kl_loss.mean()
				kl_grads = torch.autograd.grad(kl_loss, model.parameters(), create_graph=True)
				A = torch.cat([grad.view(-1) for grad in kl_grads])

			loss = get_loss()
			model.zero_grad()
			grads = self.flatten(torch.autograd.grad(loss, model.parameters()))
			loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
			
			self.iters+=1
			if torch.norm(grads)<self.eps or torch.norm(grads).isnan():
				break

			if self.first:
				t = min(1., 1./grads.abs().sum())
				sstar = -t*grads
				self.prev_flat_grad = grads
				self.vk = sstar
				self.prev_theta = get_flat_params_from(model)
				if sstar.any().isnan():
					break

			else:

				sstar = self.LSR1(grads, A, kl_loss)

			new_params = get_flat_params_from(model) + sstar
			set_flat_params_to(model, new_params)
			loss = get_loss()
			model.zero_grad()
			loss.backward(retain_graph=True)

			grads = self.gather_flat_grad(model)


			y = grads - self.prev_flat_grad

			self.mu = 0.9*self.mu
			
			if self.first:
				self.S = sstar.unsqueeze(1)
				self.Y = y.unsqueeze(1)
				self.SS = sstar.dot(sstar)[None, None]
				self.SY = sstar.dot(y)[None, None]
				self.YY = y.dot(y)[None,None]
				self.first = False

			elif self.S.shape[1]<2*self.maxhist:
				self.SY = torch.vstack((torch.hstack((self.SY, self.S.T @ y.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ self.Y , sstar.unsqueeze(1).T @ y.unsqueeze(1)))))
				self.SS = torch.vstack((torch.hstack((self.SS, self.S.T @ sstar.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ self.S , sstar.unsqueeze(1).T @ sstar.unsqueeze(1)))))
				self.YY = torch.vstack((torch.hstack((self.YY, self.Y.T @ y.unsqueeze(1))), torch.hstack((y.unsqueeze(1).T @ self.Y , y.unsqueeze(1).T @ y.unsqueeze(1)))))
				self.S = torch.cat([self.S, sstar.unsqueeze(1)], axis=1)
				self.Y = torch.cat([self.Y, y.unsqueeze(1)], axis=1)

			else:
				self.SY = torch.vstack((torch.hstack((self.SY[1:,1:], self.S[:,1:].T @ y.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ self.Y[:,1:] , sstar.unsqueeze(1).T @ y.unsqueeze(1)))))
				self.SS = torch.vstack((torch.hstack((self.SS[1:,1:], self.S[:,1:].T @ sstar.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ self.S[:,1:] , sstar.unsqueeze(1).T @ sstar.unsqueeze(1)))))
				self.YY = torch.vstack((torch.hstack((self.YY[1:,1:], self.Y[:,1:].T @ y.unsqueeze(1))), torch.hstack((y.unsqueeze(1).T @ self.Y[:,1:] , y.unsqueeze(1).T @ y.unsqueeze(1)))))
				self.S = torch.cat([self.S[:,1:], sstar.unsqueeze(1)], axis=1)
				self.Y = torch.cat([self.Y[:,1:], y.unsqueeze(1)], axis=1)

			self.prev_flat_grad = grads

			if self.verbose:
				print('------------------------------------------------------------')
				print('Iteration: {} Loss: {} Gradient norm: {} '.format(self.iters, loss.item(), torch.norm(self.gather_flat_grad(model))))
				print('Mu: {} rhok: {} gamma: {}'.format(self.mu, self.rhok, self.gamma))
				print('------------------------------------------------------------')
		
	
		return loss

	def LSR1(self,flat_grad, Avec, constraint, gammaIn=1):
		try:
			A = torch.tril(self.SY) + torch.tril(self.SY,-1).T
			B = self.SS
			if torch.isnan(A).any() or torch.isnan(B).any():
				set_trace()

			v = torch.from_numpy(sl.eigh(A.detach().cpu().numpy(),B.detach().cpu().numpy(), eigvals_only=True))
			eABmin = min(v)
			if(eABmin>0):
				self.gamma = max(0.5*eABmin, 1e-6)
			else:
				self.gamma = min(1.5*eABmin, -1e-6)

		except np.linalg.LinAlgError:	
			self.gamma=gammaIn

		V = Avec*(self.z + 1/(1+self.eps)*(self.mu/self.s - self.z - (constraint- self.s)))



		Psi = torch.hstack([self.S , 1/self.gamma*self.Y])
		R = torch.tril(self.S.T @ self.Y)
		D = torch.diag_embed(torch.diag(self.S.T @ self.Y))

		TL = torch.inverse(R).T @ (D + 1/self.gamma*self.Y.T @ self.Y) @ torch.inverse(R)
		TR = -torch.inverse(R)
		BL = -torch.inverse(R)
		
		UpperRHS = self.S.T @ V
		LowerRHS = self.gamma * self.Y.T @ V
		try:
			mid = torch.hstack([torch.vstack([TL, BL]), torch.vstack([TR, torch.zeros(TR.shape)])])
		except:
			from pdb import set_trace
			set_trace()

		Psiv = Psi.T @ V 
		midPsiT = mid @ Psiv
		PsimidPsiTv = Psi @ mid @ Psiv

		HV = self.gamma * V + PsimidPsiTv

		PsiAvec = Psi.T @ Avec
		midPsiT = mid @ PsiAvec
		PsimidPsiTAvec = Psi @ mid @ PsiAvec
		HAtilda = self.gamma * Avec + PsimidPsiTAvec
		step = HV - 1/Avec.dot(HAtilda) *((HAtilda) * HAtilda.dot(V))
		return step

	def lmarquardt(self,  D, g_parallel, C_parallel, g, U_par, alphastar, sstar, closure, model, pflag):
		if pflag:
			q = g_parallel.T @ (C_parallel**2 @ D * 0.5 - C_parallel) @ g_parallel + 0.5*(self.gamma*(alphastar)**2 - 2*alphastar)*torch.norm(g - U_par @ g_parallel)**2
			m =  q + self.mu/3*(torch.norm(C_parallel @ g_parallel, p=3)**3+ alphastar**3*torch.norm(g - U_par @ g_parallel)**3) 
		else:
			q = g_parallel.T @ (C_parallel**2 @ D * 0.5 - C_parallel) @ g_parallel
			m = q + self.mu/3*(torch.norm(C_parallel @ g_parallel, p=3)**3)

		x_init = get_flat_params_from(model)
		f1 = float(closure())
		f2 = directional_evaluate(closure, model, x_init,1, sstar)
		set_flat_params_to(model, x_init)
		return (f1 - f2)/(-m)

	def linesearch(self, theta, f, sstar):
		
		return theta


	def cubicReg(self, D, g_parallel, C_parallel, g, U_par, alphastar, sstar, closure, model, pflag):
		self.rhok = self.lmarquardt(D, g_parallel, C_parallel, g, U_par, alphastar, sstar, closure, model, pflag)
		
		if self.rhok > self.eta1:
			flag = True
		else:
			flag = False

		if self.rhok >self.eta2:
			self.mu = 0.5 * self.mu
		elif self.rhok > self.eta1 and self.rhok<self.eta2:
			self.mu  = 0.5*(self.mu + self.mu*self.gamma1)
		else:
			self.mu = 0.5*(self.gamma1 + self.gamma2)*self.mu
		
		return flag


	def trustlmarquardt(self, g, sstar, closure, model):

		m = g.T @ sstar

		x_init = get_flat_params_from(model)
		f1 = float(closure())
		f2 = directional_evaluate(closure, model, x_init,1, sstar)
		set_flat_params_to(model, x_init)
		return (f1 - f2)/(-m)

	def trustRegion(self, D, g_parallel, C_parallel, g, U_par, alphastar, sstar, gamma, closure, model, pflag):
		rhok = self.trustlmarquardt(g, sstar, closure, model)
		if rhok< self.tau2:
			self.delta = min(self.delta*self.eta1, self.eta2*torch.norm(sstar).item())
			flag = False

		else:

			flag = True
			if rhok>=self.tau3 and torch.norm(sstar)>= self.eta3*self.delta:
				self.detla = self.eta4*self.delta

		return flag





def linesearch(model,
			   f,
			   x,
			   fullstep,
			   expected_improve_rate,
			   max_backtracks=10,
			   accept_ratio=.1):
	fval = f(True).data
	for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
		xnew = x + stepfrac * fullstep
		set_flat_params_to(model, xnew)
		newfval = f(True).data
		actual_improve = fval - newfval
		expected_improve = expected_improve_rate * stepfrac
		ratio = actual_improve / expected_improve

		if ratio.item() > accept_ratio and actual_improve.item() > 0:
			return True, xnew
	return False, x
