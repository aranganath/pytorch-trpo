import numpy as np

import torch
from torch.autograd import Variable
from utils import *
import scipy.linalg as sl
from pdb import set_trace
from sys import stdout

class ARCLSR1(object):
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
		self.mu = 1
		
		self.eta1 = 0.25
		self.eta2 = 0.50
		self.eta3 = 0.75
		self.eta4 = 2
		self.gamma1 = 1
		self.gamma2 = 2
		self.gamma = 1

		# Decay parameters
		self.k = 0.1
		self.k_lower_bound = 0.01
		self.epoch_count = 0
		self.momentum = 0.9
		self.rhok = None
		self.delta = 10
		self.method = 'cubic'
		self.tau1 = 0.1
		self.tau2 = 0.2
		self.tau3 = 0.6
		self.verbose = verbose
		self.eps = 1e-8


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
		for it in range(self.maxiters):
			kl_loss = get_kl()
			kl_loss = kl_loss.mean()

			loss = get_loss()
			model.zero_grad()
			loss.backward(retain_graph=True)
			grads = self.gather_flat_grad(model)
			if torch.norm(grads)<self.eps or torch.norm(grads).isnan():
				break

			if self.verbose:
				print('------------------------------------------------------------')
				print('Iteration: {} Loss: {} Gradient norm: {} '.format(it, loss.item(), torch.norm(self.gather_flat_grad(model))))
				print('Mu: {} rhok: {} gamma: {}'.format(self.mu, self.rhok, self.gamma))
				print('------------------------------------------------------------')
			if self.first:
				t = min(1., 1./grads.abs().sum())
				sstar = -t*grads
				self.prev_flat_grad = grads
				self.vk = sstar
				if sstar.any().isnan():
					break

				flag = True

			else:
				# We have one step. let's use it
				D, g_parallel, C_parallel, U_par, alphastar, sstar, pflag = self.LSR1(self.S, self.SS, self.YY, self.SY, self.Y, grads)
				if sstar.any().isnan():
					break


				# Make sure the direction of descent lies within the trust region
				if self.method == 'cubic' and not self.first:
					#if torch.norm(self.vk).item() > 0:
					#	self.vk = self.momentum*min(1.0, 1/(torch.norm(self.vk).item()*self.mu)) * self.vk
					#	sstar = min(1.0, self.k/torch.norm(self.vk + sstar).item())*(sstar + self.vk)

					flag = self.cubicReg(D, g_parallel, C_parallel, grads, U_par, alphastar, sstar, get_loss, model, pflag)
					# else:
					# 	self.vk = sstar
					# flag = self.cubicReg(D, g_parallel, C_parallel, grads, U_par, alphastar, sstar, gamma, get_loss, model, pflag)

				if self.method == 'trust' and not self.first:
					self.vk = self.momentum*min(1.0, self.delta/torch.norm(self.vk).item()) * self.vk
					sstar = min(1.0, self.delta/torch.norm(self.vk + sstar).item())*(sstar + self.vk)

					if torch.norm(sstar)> self.delta:
						sstar = sstar/torch.norm(sstar)*self.delta

					flag = self.trustRegion(D, g_parallel, C_parallel, grads, U_par, alphastar, sstar, gamma, get_loss, model, pflag)

			new_params = get_flat_params_from(model) + sstar
			set_flat_params_to(model, new_params)
			loss = get_loss()
			model.zero_grad()
			loss.backward(retain_graph=True)

			grads = self.gather_flat_grad(model)


			if flag:
				y = grads - self.prev_flat_grad
				if self.first:
					self.S = sstar.unsqueeze(1)
					self.Y = y.unsqueeze(1)
					self.SS = sstar.dot(sstar)[None, None]
					self.SY = sstar.dot(y)[None, None]
					self.YY = y.dot(y)[None,None]
					self.first = False

				elif self.S.shape[1]<self.maxhist:
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
		
	
		return loss

	def LSR1(self, S, SS, YY, SY, Y, flat_grad, gammaIn=1):

		try:
			A = torch.tril(SY) + torch.tril(SY,-1).T
			B = SS
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

		Psi = Y - self.gamma*S
		PsiPsi = Psi.T @ Psi
		R=torch.linalg.cholesky(PsiPsi.transpose(-2,-1).conj()).transpose(-2,-1).conj()
		Q = Psi @ torch.inverse(R)
		if R.detach().any().isnan():
			set_trace()

		invM = torch.tril(SY) + torch.tril(SY, -1).T - self.gamma*SS
		invM = 0.5*(invM + invM.T)

		try:
			RMR = R @ torch.inverse(invM) @ R.T
		except RuntimeError:
			from pdb import set_trace
			set_trace()

		RMR = 0.5*(RMR + RMR.T)
		D,P = torch.linalg.eigh(RMR)
		U_par = Q.detach() @ P.detach()
		if len(U_par.shape)==1:
			U_par = U_par.unsqueeze(1)

		if D.detach().any().isnan():
			set_trace()

		#Compute lambda as in equation (7)
		#set_trace()
		try:
			g_parallel = U_par.T @ flat_grad
		except:
			from pdb import set_trace
			set_trace()
		C_parallel = []

		gperp = flat_grad - U_par @ g_parallel
		for i,lam in enumerate(D):
			c_i = 2/(lam + torch.sqrt(lam**2 + 4*self.mu*torch.abs(g_parallel[i])))
			C_parallel.append(c_i)

		C_parallel = torch.diag(torch.stack(C_parallel).reshape(-1))


		if torch.norm(gperp) < 1e-8:
			#The solution only depends on sstar_parallel
			sstar_parallel = -C_parallel @ g_parallel
			sstar =  U_par @ sstar_parallel
			if sstar.isnan().any():
				set_trace()
			alphastar = 0
			return D, g_parallel, C_parallel, U_par, alphastar, sstar, False
		
		

		alphastar = 2/(self.gamma + torch.sqrt(self.gamma**2 + 4*self.mu*torch.norm(gperp)))
		sstar = -alphastar * flat_grad + U_par @ (alphastar * torch.eye(S.shape[1]) - C_parallel) @ g_parallel
		if sstar.isnan().any():
			from pdb import set_trace
			set_trace()

		return D, g_parallel, C_parallel, U_par, alphastar, sstar, True

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
