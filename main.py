import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from RL_ARCLSR1 import ARCLSR1
from InteriorPointMethod import InteriorPointMethod
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
import pickle as pkl
import os
from sys import stdout
import time
import matplotlib.pyplot as plt
from pdb import set_trace

torch.set_default_tensor_type('torch.cuda.DoubleTensor')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v2", metavar='G',help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G', help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G', help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G', help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G', help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=15000, metavar='N', help='random seed (default: 1)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

def select_action(state, policy_net, value_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state).to(device))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch, policy_net, value_net):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state)) # cast list to np array
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().cpu().numpy(), get_flat_grad_from(value_net).data.double().cpu().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().cpu().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))

        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        # Original trpo loss function
        # action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        action_loss1 = Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        # print('Ratio: {}'.format(torch.exp(log_prob - Variable(fixed_log_prob))[torch.exp(log_prob - Variable(fixed_log_prob))<0]))
        eps = 0.5
        action_loss2 = Variable(advantages) * torch.clip(torch.exp(log_prob - Variable(fixed_log_prob)), min=1-eps, max=1+eps)
        action_loss = -torch.min(action_loss1, action_loss2)
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    if opt =='ARCLSR1': # Our Proposed
        optimize.arclsr1(policy_net, get_loss, get_kl, args.max_kl, args.damping, environment)

    if opt =='trpo': # Existing Solution
        trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

    if opt == 'InteriorPointMethod':
        InteriorOptimize.computeStep(policy_net, get_loss, get_kl)



envs = ['Swimmer-v4'] # Look up other enviornments ,
opts = ['ARCLSR1', 'trpo'] #

optimize = ARCLSR1(maxhist = 100, maxiters = 100, verbose=True)

for environment in envs:
    for opt in opts:
        print('Environment: {}, Optimizer: {}'.format(environment, opt), flush=True)
        env = gym.make(environment)

        num_inputs = env.observation_space.shape[0]

        # set_trace()

        num_actions = env.action_space.shape[0]

        print(num_inputs, num_actions)

        env.reset(seed=args.seed)
        torch.manual_seed(args.seed)

        policy_net = Policy(num_inputs, num_actions).to(device)
        value_net = Value(num_inputs).to(device)
        
        total_rewards = []

        running_state = ZFilter((num_inputs,), clip=5)
        running_reward = ZFilter((1,), demean=False, clip=10)

        episodes = 500 # Min 500; Max 5000

        for i_episode in count(1): # 1, 2, . . . , episodes; Auto-increments, but must break out of loop

            print(f"\nEpisode: {i_episode}")

            # if i_episode == 10:
                # break

            memory = Memory()

            num_steps = 0
            reward_batch = 0
            num_episodes = 0

            print(f"batch size: {args.batch_size}")
            print(f"num_steps: {num_steps}\n")

            while num_steps < args.batch_size:

                state = env.reset()

                # set_trace()

                state = running_state(state[0]) # 376 states in HumanoidStandup-v4

                reward_sum = 0

                for t in range(10000): # Don't infinite loop while learning
                    action = select_action(state, policy_net, value_net)

                    action = action.data[0].cpu().numpy()

                    #time.sleep(0.002)
                    next_state, reward, done, _, _ = env.step(action)

                    reward_sum += reward

                    next_state = running_state(next_state)

                    mask = 1
                    if done:
                        mask = 0

                    memory.push(state, np.array([action]), mask, next_state, reward)

                    if args.render:
                        env.render()
                    if done:
                        break

                    state = next_state

                print(f"reward sum: {reward_sum}\n")
                num_steps += (t-1)
                print(num_steps, args.batch_size)
                num_episodes += 1
                reward_batch += reward_sum

            reward_batch /= num_episodes
            batch = memory.sample()
            update_params(batch, policy_net, value_net)

            if i_episode % args.log_interval == 0:

                print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                    i_episode, reward_sum, reward_batch))
                total_rewards.append(reward_batch)

            # plt.plot(total_rewards, i_episode)
            # plt.show()
            if i_episode == episodes:
                break



        if not os.path.isdir(environment):
            os.mkdir(environment)

        print('Saving at location ./'+environment+'/'+opt+str(episodes)+'.pkl')

        with open('./'+environment+'/'+opt+str(episodes)+'.pkl','wb') as f:
        	pkl.dump(total_rewards, f, protocol=pkl.HIGHEST_PROTOCOL)
