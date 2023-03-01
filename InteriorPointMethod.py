import numpy as np

import torch
from torch.autograd import Variable
from utils import *
import scipy.linalg as sl
from pdb import set_trace
from sys import stdout
from pdb import set_trace


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
		self.mu = 1
		self.gamma = 1

		self.rhok = None
		self.verbose = verbose
		
		self.iters = 0
		self.prev_theta = None
		self.z = torch.tensor(1e-4)
		self.s = torch.tensor(10.)
		self.eps = self.z/self.s
		self.epsilon = 1e-8


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

	def computeHv(self, vector):
		Psi = torch.hstack([self.S ,self.gamma*self.Y])
		R = torch.tril(self.SY)
		D = torch.diag_embed(torch.diag(self.SY))
		
		TL = torch.inverse(R).T @ (D + self.gamma*self.YY) @ torch.inverse(R)
		TR = -torch.inverse(R).T
		BL = -torch.inverse(R)

		try:
			mid = torch.hstack([torch.vstack([TL, BL]), torch.vstack([TR, torch.zeros(TR.shape)])])
		except:
			set_trace()

		Psiv = Psi.T @ vector
		midPsiT = mid @ Psiv
		PsimidPsiTv = Psi @ midPsiT
		return self.gamma * vector + PsimidPsiTv


	def LineSearchSQP(self, f, constraint, p, model):
		alpha = 0.9
		tau = 1.
		eta = 1.
		phi = lambda theta, mu: f() + self.mu * torch.abs(constraint().mean())
		theta = get_flat_params_from(model)
		set_trace()
		D = lambda func, pk: torch.autograd.grad(func, Variable(p))
		phi0 = phi(theta, self.mu)
		phi1 = phi(theta + alpha*p, self.mu)
		if phi1 <= phi0 + eta * alpha * D(phi, p):
			set_trace()

		

	def lbfgs(self, model, get_loss, get_kl, max_kl, damping, environment):
		# We use the gradient from the traditional loss function
		# But we use the hessian approximation using the kl-divergence
		# How do we do it ?
		# For the kl divergence, we will use the hessian
		for it in range(self.maxiters):
			if self.prev_theta is not None:
				kl_loss = get_kl()
				kl_loss = kl_loss.mean()
				kl_grads = torch.autograd.grad(kl_loss, model.parameters(), create_graph=False)
				A = torch.cat([grad.view(-1) for grad in kl_grads])

			loss = get_loss()
			model.zero_grad()
			grads = self.flatten(torch.autograd.grad(loss, model.parameters()))
			loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
			
			self.iters+=1
			if torch.norm(grads)<self.epsilon or torch.norm(grads).isnan():
				set_flat_params_to(model, self.prev_theta)
				set_trace()
				break

			

			if self.first:
				t = min(1., 1./grads.abs().sum())
				sstar = -t*grads
				self.prev_flat_grad = grads
				self.prev_theta = get_flat_params_from(model)
				if sstar.any().isnan():
					break

			else:

				sstar, ds, dz = self.LBFGS(grads, A, kl_loss)
			old_params = get_flat_params_from(model)
			new_params = get_flat_params_from(model) + sstar
			set_flat_params_to(model, new_params)
			loss = get_loss()
			model.zero_grad()
			loss.backward(retain_graph=True)

			grads = self.gather_flat_grad(model)
			
			set_flat_params_to(model, old_params)
			loss = get_loss()
			model.zero_grad()
			loss.backward(retain_graph=True)
			set_trace()

			y = grads - self.prev_flat_grad

			if not self.first:
				if sstar.dot(y) >= 0.2*sstar.dot(self.computeHv(sstar)):
					theta = 1
				else:
					hessvec = sstar.dot(self.computeHv(sstar))
					theta = 0.8* hessvec/(hessvec - sstar.dot(y))

				y = theta*y + (1-theta)*self.computeHv(sstar)

				if y.dot(sstar)>1e-10:
			
					if self.S.shape[1]<self.maxhist:
						self.SY = torch.vstack((torch.hstack((self.SY, self.S.T @ y.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ self.Y , sstar.unsqueeze(1).T @ y.unsqueeze(1)))))
						self.SS = torch.vstack((torch.hstack((self.SS, self.S.T @ sstar.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ self.S , sstar.unsqueeze(1).T @ sstar.unsqueeze(1)))))
						self.YY = torch.vstack((torch.hstack((self.YY, self.Y.T @ y.unsqueeze(1))), torch.hstack((y.unsqueeze(1).T @ self.Y , y.unsqueeze(1).T @ y.unsqueeze(1)))))
						self.S = torch.cat([self.S, sstar.unsqueeze(1)], axis=1)
						self.Y = torch.cat([self.Y, y.unsqueeze(1)], axis=1)
						self.s += ds
						self.z += dz

					else:
						self.SY = torch.vstack((torch.hstack((self.SY[1:,1:], self.S[:,1:].T @ y.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ self.Y[:,1:] , sstar.unsqueeze(1).T @ y.unsqueeze(1)))))
						self.SS = torch.vstack((torch.hstack((self.SS[1:,1:], self.S[:,1:].T @ sstar.unsqueeze(1))), torch.hstack((sstar.unsqueeze(1).T @ self.S[:,1:] , sstar.unsqueeze(1).T @ sstar.unsqueeze(1)))))
						self.YY = torch.vstack((torch.hstack((self.YY[1:,1:], self.Y[:,1:].T @ y.unsqueeze(1))), torch.hstack((y.unsqueeze(1).T @ self.Y[:,1:] , y.unsqueeze(1).T @ y.unsqueeze(1)))))
						self.S = torch.cat([self.S[:,1:], sstar.unsqueeze(1)], axis=1)
						self.Y = torch.cat([self.Y[:,1:], y.unsqueeze(1)], axis=1)
						self.s += ds
						self.z += dz

					set_flat_params_to(model, new_params)
					
					self.prev_flat_grad = grads
					self.eps = self.z/self.s


			else:
				self.S = sstar.unsqueeze(1)
				self.Y = y.unsqueeze(1)
				self.SS = sstar.dot(sstar)[None, None]
				self.SY = sstar.dot(y)[None, None]
				self.YY = y.dot(y)[None,None]
				self.first = False

				set_flat_params_to(model, new_params)
				
				self.prev_flat_grad = grads


			self.mu = 0.9*self.mu

				
			if self.verbose:
				print('------------------------------------------------------------')
				print('Iteration: {} Loss: {} Gradient norm: {} '.format(self.iters, loss.item(), torch.norm(self.gather_flat_grad(model))))
				print('Mu: {} rhok: {} gamma: {} s: {} z: {}'.format(self.mu, self.rhok, self.gamma, self.s, self.z))
				print('------------------------------------------------------------')
			
			# if torch.norm(self.gather_flat_grad(model)) < 1e-4:
			# 	break
	
		return loss

	def LBFGS(self,flat_grad, Avec, constraint):
		self.gamma = self.Y[:,-1].dot(self.S[:,-1])/self.Y[:,-1].dot(self.Y[:,-1])
		V = -flat_grad + self.mu/self.s * Avec - Avec * self.eps *(constraint - self.s)
		HV = self.computeHv(V)
		ATilda = Avec*torch.sqrt(self.eps)
		HAtilda = self.computeHv(ATilda)
		AHV = ATilda.dot(HV)
		numerator = HAtilda * AHV
		denominator = 1 + ATilda.dot(HAtilda)
		step = HV - numerator/(denominator)
		dz = self.mu / self.s - self.z - self.eps*(constraint - self.s + Avec.dot(step))
		ds = constraint - self.s + Avec.dot(step)
		return step, ds, dz