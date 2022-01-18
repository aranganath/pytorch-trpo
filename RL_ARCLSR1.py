import numpy as np

import torch
from torch.autograd import Variable
from utils import *
import scipy.linalg as sl

class ARCLSR1(object):
    def __init__(self, maxiters=10, maxhist=10):
        self.S = None
        self.Y = None
        self.SS = None
        self.YY = None
        self.SY = None
        self.prev_flat_grad = []
        self.flat_grad = []
        self.lr = 1e-1
        self.maxiters = 1
        self.maxhist = 1
        self.first = True
        self.mu = 100
        self.eta1 = 0.05
        self.eta2 = 0.15
        self.gamma1 = 1
        self.gamma2 = 2

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


    def arclsr1(self, model, get_loss, get_kl, max_kl, damping):

        # We use the gradient from the traditional loss function
        # But we use the hessian approximation using the kl-divergence
        # How do we do it ?
        # Fot the kl divergence, we will use the hessian
        for it in range(self.maxiters):
            kl_loss = get_kl()
            kl_loss = kl_loss.mean()

            loss = get_loss()
            kl_grads = self.flatten(torch.autograd.grad(kl_loss, model.parameters(), create_graph=True))
            grads = self.flatten(torch.autograd.grad(loss, model.parameters(), create_graph=True))
            if self.first:
                t = min(1., 1./grads.abs().sum())*self.lr
                s = t*grads
                self.prev_flat_grad = grads
                flag = True

            else:
                # We have one step. let's use it
                D, g_parallel, C_parallel, U_par, alphastar,sstar, gamma, pflag = self.LSR1(self.S, self.SS, self.YY, self.SY, self.Y, grads, 1)
                flag = self.cubicReg(D, g_parallel, C_parallel, grads, U_par, alphastar, sstar, gamma, get_loss, model, pflag)

            if flag:
                new_params = get_flat_params_from(model) + s
                set_flat_params_to(model, new_params)
                # Update parameters
                
                grads = self.flatten(torch.autograd.grad(loss, model.parameters(), create_graph=True))
                y = grads - self.prev_flat_grad
                

                if self.first:

                    self.S = s.unsqueeze(1)
                    self.Y = y.unsqueeze(1)
                    self.SS = s.dot(s)[None, None]
                    self.SY = s.dot(y)[None, None]
                    self.YY = y.dot(y)[None,None]
                    self.first = False

                elif self.S.shape[1]<self.maxhist:

                    self.SY = torch.vstack((torch.hstack((self.SY, self.S.T @ y.unsqueeze(1))), torch.hstack((s.unsqueeze(1).T @ self.Y , s.unsqueeze(1).T @ y.unsqueeze(1)))))
                    self.SS = torch.vstack((torch.hstack((self.SS, self.S.T @ s.unsqueeze(1))), torch.hstack((s.unsqueeze(1).T @ self.S , s.unsqueeze(1).T @ s.unsqueeze(1)))))
                    self.YY = torch.vstack((torch.hstack((self.YY, self.Y.T @ y.unsqueeze(1))), torch.hstack((y.unsqueeze(1).T @ self.Y , y.unsqueeze(1).T @ y.unsqueeze(1)))))
                    self.S = torch.cat([self.S, s.unsqueeze(1)], axis=1)
                    self.Y = torch.cat([self.Y, y.unsqueeze(1)], axis=1)

                else:

                    self.SY = torch.vstack((torch.hstack((self.SY[1:,1:], self.S[:,1:].T @ y.unsqueeze(1))), torch.hstack((s.unsqueeze(1).T @ self.Y[:,1:] , s.unsqueeze(1).T @ y.unsqueeze(1)))))
                    self.SS = torch.vstack((torch.hstack((self.SS[1:,1:], self.S[:,1:].T @ s.unsqueeze(1))), torch.hstack((s.unsqueeze(1).T @ self.S[:,1:] , s.unsqueeze(1).T @ s.unsqueeze(1)))))
                    self.YY = torch.vstack((torch.hstack((self.YY[1:,1:], self.Y[:,1:].T @ y.unsqueeze(1))), torch.hstack((y.unsqueeze(1).T @ self.Y[:,1:] , y.unsqueeze(1).T @ y.unsqueeze(1)))))
                    self.S = torch.cat([self.S[:,1:], s.unsqueeze(1)], axis=1)
                    self.Y = torch.cat([self.Y[:,1:], y.unsqueeze(1)], axis=1)

                self.prev_flat_grad = grads
            
        return loss

    def LSR1(self, S, SS, YY, SY, Y, flat_grad, gammaIn):

        try:
            A = torch.tril(SY) + torch.tril(SY,-1).T
            B = SS
            if torch.isnan(A).any() or torch.isnan(B).any():
                set_trace()

            v = torch.from_numpy(sl.eigh(A.detach().cpu().numpy(),B.detach().cpu().numpy(), eigvals_only=True))
            eABmin = min(v)
            if(eABmin>0):
                gamma = max(0.5*eABmin, 1e-6)
            else:
                gamma = min(1.5*eABmin, -1e-6)

        except np.linalg.LinAlgError:   
            gamma=gammaIn

        Psi = Y - gamma*S
        PsiPsi = YY - gamma*(SY + SY.T) + gamma**2*SS
        R=torch.linalg.cholesky(PsiPsi.transpose(-2,-1).conj()).transpose(-2,-1).conj()
        Q = Psi @ torch.inverse(R)
        if R.detach().any().isnan():
            set_trace()

        invM = torch.tril(SY) + torch.tril(SY, -1).T - gamma*SS
        invM = 0.5*(invM + invM.T)

        try:
            RMR = R @ torch.inverse(invM) @ R.T
        except RuntimeError:
            from pdn import set_trace
            set_trace()

        RMR = 0.5*(RMR + RMR.T)
        D,P = torch.linalg.eigh(RMR)
        U_par = Q.detach() @ P.detach()
        if len(U_par.shape)==1:
            U_par = U_par.unsqueeze(1)

        if D.detach().any().isnan():
            set_trace()

        #Compute lambda as in equation (7)
        g_parallel = U_par.T @ flat_grad
        C_parallel = []

        gperp = flat_grad - U_par @ g_parallel
        for i,lam in enumerate(D):
            c_i = 2/(lam + torch.sqrt(lam**2 + 4*0.1*torch.abs(g_parallel[i])))
            C_parallel.append(c_i)

        C_parallel = torch.diag(torch.stack(C_parallel).reshape(-1))


        if torch.norm(gperp) < 1e-15:
            #The solution only depends on sstar_parallel
            sstar_parallel = -C_parallel @ g_parallel
            sstar =  U_par @ sstar_parallel
            if sstar.isnan().any():
                set_trace()
            alphastar = 0
            return D, g_parallel, C_parallel, U_par, alphastar, sstar, gamma, False
        
        

        alphastar = 2/(gamma + torch.sqrt(gamma**2 + 4*0.1*torch.norm(gperp)))
        sstar = -alphastar * flat_grad + U_par @ (alphastar * torch.eye(S.shape[1]) - C_parallel) @ g_parallel
        if sstar.isnan().any():
            set_trace()

        return D, g_parallel, C_parallel, U_par, alphastar, sstar, gamma, True

    def lmarquardt(self,  D, g_parallel, C_parallel, g, U_par, alphastar, sstar, gamma, closure, model, pflag):
        if pflag:
    
            q = g_parallel.T @ (C_parallel**2 @ D * 0.5 - C_parallel) @ g_parallel + 0.5*(gamma*(alphastar)**2 - 2*alphastar)*torch.norm(g - U_par @ g_parallel)**2
            m =  q + self.mu/3*(torch.norm(C_parallel @ g_parallel, p=3)**3+ alphastar**3*torch.norm(g - U_par @ g_parallel)**3) 
        else:
            q = g_parallel.T @ (C_parallel**2 @ D * 0.5 - C_parallel) @ g_parallel
            m = q + self.mu/3*(torch.norm(C_parallel @ g_parallel, p=3)**3)

        x_init = get_flat_params_from(model)
        f1 = float(closure())
        f2 = directional_evaluate(closure, model, x_init,1, sstar)
        set_flat_params_to(model, x_init)
        return (f1 - f2)/(-m)




    def cubicReg(self, D, g_parallel, C_parallel, g, U_par, alphastar, sstar, gamma, closure, model, pflag):
        rhok = self.lmarquardt(D, g_parallel, C_parallel, g, U_par, alphastar, sstar, gamma, closure, model, pflag)
        
        if rhok > self.eta1:
            flag = True
        else:
            flag = False

        if rhok >self.eta2:
            self.mu = 0.5 * self.mu
        elif rhok > self.eta1 and rhok<self.eta2:
            self.mu  = 0.5*(self.mu + self.mu*self.gamma1)
        else:
            self.mu = 0.5*(self.gamma1 + self.gamma2)*self.mu

        return flag

