from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
from spinup.utils.run_utils import set_mujoco; set_mujoco(); import gym
import time
import spinup.algos.pytorch.egl.core as core
from spinup.utils.logx import EpochLogger
import math
import torch.nn.functional as F
from tqdm import tqdm
import torch.autograd as autograd

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device):

        self.device = device
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k,v in batch.items()}


def repeat_and_reshape(x, n):

    x_expand = x.repeat(n, *[1] * len(x.shape))
    x_expand = x_expand.view(n * len(x), -1)

    return x_expand


def ball_explore(a1, n_explore, eps):

    b, act_dim = a1.shape
    a1 = a1.unsqueeze(0)
    x = torch.zeros_like(a1).normal_()
    mag = torch.zeros(n_explore, b, 1, device=a1.device).uniform_()

    x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    a2 = a1 + eps * math.sqrt(act_dim) * mag * x

    return a2


def eglu(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, eps=0.2, n_explore=32, device='cuda'):

    device = torch.device(device)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing EGL mean-gradient-losses
    def compute_loss_g(data):

        o, a1, r, o_tag, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        a2 = ball_explore(a1, n_explore, eps)

        a2 = a2.view(n_explore * len(r), act_dim)
        o_expand = repeat_and_reshape(o, n_explore)

        # Bellman backup for Q functions
        with torch.no_grad():

            q1 = ac.q1(o_expand, a2)
            q2 = ac.q2(o_expand, a2)
            q_dither = torch.min(q1, q2)

            # Target actions come from *current* policy
            a_tag, logp_a_tag = ac.pi(o_tag)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o_tag, a_tag)
            q2_pi_targ = ac_targ.q2(o_tag, a_tag)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            q_anchor = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a_tag)

            q_anchor = repeat_and_reshape(q_anchor, n_explore).squeeze(-1)

        a1_in = autograd.Variable(a1.data, requires_grad=True)
        q1 = ac.q1(o, a1_in)
        q2 = ac.q2(o, a1_in)
        qa = torch.min(q1, q2).unsqueeze(-1)
        geps = autograd.grad(outputs=qa, inputs=a1_in, grad_outputs=torch.cuda.FloatTensor(qa.size()).fill_(1.),
                                  create_graph=False, retain_graph=True, only_inputs=True)[0]

        geps = repeat_and_reshape(geps, n_explore)
        a1 = repeat_and_reshape(a1, n_explore)

        geps = (geps * (a2 - a1)).sum(-1)
        # l1 loss against Bellman backup

        loss_g = F.smooth_l1_loss(geps, q_dither - q_anchor)

        # Useful info for logging
        g_info = dict(GVals=geps.flatten().detach().cpu().numpy())

        return loss_g, g_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()

        # Next run one gradient descent step for the mean-gradient
        loss_g, g_info = compute_loss_g(data)
        # Record things
        logger.store(LossG=loss_g.item(), **g_info)

        q_optimizer.zero_grad()

        loss_q, q_info = compute_loss_q(data)
        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        loss_q = loss_q + loss_g

        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in ac.geps.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in ac.geps.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in tqdm(range(total_steps)):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_explore', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--exp_name', type=str, default='egl')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    exp_name = f"{args.env.split('-')[0]}_{args.exp_name}"
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    print("EGL Experiment")
    eglu(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, eps=args.eps, n_explore=args.n_explore,
        device=args.device, batch_size=args.batch_size)
