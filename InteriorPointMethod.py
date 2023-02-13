import numpy as np

import torch
from torch.autograd import Variable
from utils import *
import scipy.linalg as sl
from pdb import set_trace
from sys import stdout

class InteriorPointMethod(object):
	def __init__(self,
				maxiters=100, 
				maxhist=100, 
				method='LBFGS', 
				verbose=False):

		self.S = []
		self.Y = []
		self.sigma = 1
		self.method = method
		self.learningrate = 1e-4
		self.mu = 0.9
		self.mureduc = 0.99


	def computeStep(self, model, get_loss, constraint):
		grads = self.gather_flat_grad()

		if not self.S:
			t = min(1., 1./grads.abs().sum())
			s = -t*grads
			self.prev_flat_grad = grads
			self.S.append(s)

			from pdb import set_trace
			set_trace()

		else:
			# compute A_I
			self.model.zero_grad()
			grad = self.loss.backward(retain_graph=True)
			A_I = self.constraint.backward(retain_graph=True)

			# v = # A_I *[z + 1/(1+\Sigma) *(\mu/s - z - [c_I(\theta) - s])]

			# Now, using S and Y, compute the Quasi-Newton inverse form (not explicity doing the outer product)




	def gather_flat_grad(self):
		views = []
		for p in self.model.parameters():
			p.retain_grad()
			if p.grad is None:
				view = p.new(p.numel()).zero_()
			elif p.grad.is_sparse:
				view = p.grad.to_dense().view(-1)
			else:
				view = p.grad.view(-1)
			views.append(view)
		return torch.cat(views, axis=0)
