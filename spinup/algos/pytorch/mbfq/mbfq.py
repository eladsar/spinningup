from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
from spinup.utils.run_utils import set_mujoco;
set_mujoco();
import os
import gym
import time
import spinup.algos.pytorch.mbfq.core as core
from spinup.utils.logx import EpochLogger
from .spline_model import SparseDenseAdamOptimizer
import torch.autograd as autograd
import math
import torch.nn.functional as F
from tqdm import tqdm


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
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}


def mbfq(env_fn, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, lr=1e-3, alpha=0.2, batch_size=128, start_steps=10000,
         update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
         logger_kwargs=dict(), save_freq=1, update_factor=1, device='cuda'):

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

    # state_dim = {376: 144, 111: 64, 17: 12, 11: 8}[obs_dim[0]]
    state_dim = 128
    # Create actor-critic module and target networks
    ac = core.MLPActorCritic(state_dim, env.action_space, **ac_kwargs).to(device)
    # model = core.WorldModel(obs_dim[0], state_dim, act_dim).to(device)
    model = core.DeterministicWorldModel(obs_dim[0], 128, act_dim).to(device)

    # temporary hook
    environment = 'halfcheetah'
    root_dir = '/mnt/dsi_vol1/users/elads/data/spinningup/data'
    log_dir = os.path.join(root_dir, f'{environment}_world_model')
    model.load_state_dict(torch.load(os.path.join(log_dir, 'model')))

    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v, model.r, model.ae])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t r: %d, \t ae: %d\n' % var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_model(data):

        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        loss_model, model_info = model(o, a, r, o2, d)

        return loss_model, model_info

    # Set up function for computing SAC Q-losses
    def compute_loss_v(data):

        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # s, s2, r, d = model.gen(o, a)
        #
        # with torch.no_grad():
        #
        #     v2 = ac_targ.v(s2)
        #     g_target = r + gamma * (1 - d) * v2
        #     _, logp_pi = ac.pi(s)

        with torch.no_grad():
            s = model.get_state(o)
            s2 = model.get_state(o2)
            v2 = ac_targ.v(s2)
            g_target = r + gamma * (1 - d) * v2
            _, logp_pi = ac.pi(s)

        v = ac.v(s)

        loss_v = ((v - g_target + alpha * logp_pi) ** 2).mean()

        # Useful info for logging
        v_info = dict(VVals=v.detach().cpu().numpy())

        return loss_v, v_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']

        with torch.no_grad():
            s = model.get_state(o)

        pi, logp_pi = ac.pi(s)
        c = torch.cat([s, pi], dim=1)

        # grad_q = model.grad_q(s, pi.detach(), gamma, ac.v)
        qa = model.grad_q(c, gamma, ac.v)
        loss_pi = (alpha * logp_pi - qa).mean()

        # loss_pi = (alpha * logp_pi - (grad_q * pi).sum(-1)).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy(),
                       # GradQAmp=torch.norm(grad_q, dim=-1).detach().cpu().numpy(),
                       ActionsNorm=torch.norm(pi, dim=-1).detach().cpu().numpy(),
                       ActionsAbs=torch.abs(pi).flatten().detach().cpu().numpy(), )

        return loss_pi, pi_info

    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    v_optimizer = Adam(ac.v.parameters(), lr=lr)
    # model_optimizer = Adam(model.parameters(), lr=lr)
    model_optimizer = SparseDenseAdamOptimizer(model, dense_args={'lr': lr}, sparse_args={'lr': 10 * lr})

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):

        # First run one gradient descent step for the world-model
        # loss_model, model_info = compute_loss_model(data)
        # model_optimizer.zero_grad()
        # loss_model.backward()
        # model_optimizer.step()
        # # Record things
        # logger.store(LossModel=loss_model.item(), **model_info)

        # next run one gradient descent step for the world-model
        loss_v, v_info = compute_loss_v(data)
        v_optimizer.zero_grad()
        loss_v.backward()
        v_optimizer.step()
        # Record things
        logger.store(LossV=loss_v.item(), **v_info)

        # Next run one gradient descent step for pi.
        loss_pi, pi_info = compute_loss_pi(data)
        pi_optimizer.zero_grad()
        loss_pi.backward()
        pi_optimizer.step()
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

        s = model.get_state(torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0), batch_size=batch_size).squeeze(0)
        return ac.act(s, deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
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

        # eps = math.exp(math.log(1.) + (math.log(0.03) - math.log(1.)) * math.sin(2 * math.pi * t / 200e3))
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
        d = False if ep_len == max_ep_len else d

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
            for j in range(update_every * update_factor):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

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
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            # logger.log_tabular('LossModel', average_only=True)
            # logger.log_tabular('GradQAmp', with_min_and_max=True)
            logger.log_tabular('ActionsNorm', with_min_and_max=True)
            logger.log_tabular('ActionsAbs', with_min_and_max=True)

            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eps', type=float, default=0.4)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--exp_name', type=str, default='egl')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    exp_name = f"{args.env.split('-')[0]}_{args.exp_name}"
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    print("EGL Experiment")
    mbfq(lambda: gym.make(args.env), ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs,
         device=args.device, batch_size=args.batch_size)
