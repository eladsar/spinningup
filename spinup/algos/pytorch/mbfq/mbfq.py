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
from spinup.algos.pytorch.ppo.core import discount_cumsum
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from tqdm import tqdm


class VirtualEnv(object):

    def __init__(self, rb, model):
        super().__init__()

        self.model = model
        self.rb = rb
        self.o, self.t, self.score, self.k = None, False, 0, 0
        self.reset()

    def __bool__(self):
        return not bool(self.t)

    def step(self, a):

        training = self.model.training
        self.model.eval()

        if len(a) % 2:
            a = np.append(a, 0)

        a = torch.as_tensor(a, dtype=torch.float32, device=self.o.device).unsqueeze(0)

        with torch.no_grad():
            s = self.model.ae.encoder(self.o)

            c = torch.cat([s, a], dim=-1)
            r = float(self.model.r(c))
            d = bool(torch.bernoulli(torch.sigmoid(self.model.d(c))))
            o = self.model.sample(c=c, num_samples=1)

        self.t = d
        self.score += r
        self.s = s

        if training:
            self.model.train()

        return o.squeeze(0).cpu().numpy(), r, d, {}

    def reset(self):

        self.t, self.score, self.k = False, 0, 0

        # i = torch.randint(len(self.rb), size=(1,)).item()
        # self.o = self.rb['obs'][i].unsqueeze(0)

        sample = self.rb.sample_batch(batch_size=1)
        self.o = sample['obs']

        return self.o.squeeze(0).cpu().numpy()


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, device, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.device = device
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}


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
         steps_per_epoch=2000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, lr=1e-3, alpha=0.2, batch_size=128, start_steps=10000,
         update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, max_ep_len_ppo=50,
         logger_kwargs=dict(), save_freq=1, update_factor=1, device='cuda', lam=0.97, steps_per_ppo_update=1000,
         n_ppo_updates=1, train_pi_iters=80, target_kl=0.01, clip_ratio=0.2):

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

    state_dim = {376: 144, 111: 64, 17: 12, 11: 8}[obs_dim[0]]

    # Create actor-critic module and target networks
    ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs).to(device)

    model = core.FlowWorldModel(obs_dim[0], state_dim, act_dim+int(act_dim % 2)).to(device)

    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    ppo_buffer = PPOBuffer(obs_dim, act_dim, steps_per_ppo_update,  gamma=gamma, lam=lam, device=device)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, model])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d, \t model: %d\n' % var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_model(data):

        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        if act_dim % 2:
            a = torch.cat([a, torch.zeros(len(a), 1, device=a.device)], dim=1)

        loss, info, _ = model(o, a, r, o2, d)

        return loss, info

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
    model_optimizer = SparseDenseAdamOptimizer(model, dense_args={'lr': lr}, sparse_args={'lr': 10 * lr})

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):

        loss_model, model_info = compute_loss_model(data)
        model_optimizer.zero_grad()
        loss_model.backward()
        core.clip_grad_norm(model.parameters(), 1000)
        model_optimizer.step()

        # Record things
        logger.store(LossModel=loss_model.item(), **model_info)

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
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

        # s = model.get_state(torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0), batch_size=batch_size).squeeze(0)
        # return ac.act(o, deterministic)
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device),
                      deterministic)

    # Set up function for computing PPO policy loss
    def ppo_compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        ac.pi(obs)
        logp = ac.pi.log_prob(act, desquash=True)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, cf=clipfrac)

        return loss_pi, pi_info

    def ppo_step(o):

        with torch.no_grad():
            o = torch.as_tensor(o, dtype=torch.float32, device=device)
            a, log_pi = ac_targ.pi(o)
            q1_pi = ac.q1(o, a)
            q2_pi = ac.q2(o, a)
            q_pi = torch.min(q1_pi, q2_pi)
            v = (alpha * log_pi - q_pi).squeeze(0).cpu().numpy()

        return a.squeeze(0).cpu().numpy(), v, log_pi.squeeze(0).cpu().numpy()

    def virtual_ppo():

        venv = VirtualEnv(replay_buffer, model)
        ac_targ.pi.load_state_dict(ac.pi.state_dict())

        # Main loop: collect experience in env and update/log each epoch

        for epoch in range(n_ppo_updates):

            o, ep_ret, ep_len = venv.reset(), 0, 0

            for t in tqdm(range(steps_per_ppo_update)):

                a, v, log_pi = ppo_step(o)

                next_o, r, d, _ = venv.step(a)
                ep_ret += r
                ep_len += 1

                # save and log
                ppo_buffer.store(o, a, r, v, log_pi)
                logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len_ppo
                terminal = d or timeout
                epoch_ended = t == steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = ppo_step(o)

                    else:
                        v = 0
                    ppo_buffer.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(VirtualEpRet=ep_ret, VirtualEpLen=ep_len)
                    o, ep_ret, ep_len = env.reset(), 0, 0

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            # Perform PPO update!
            data = ppo_buffer.get()

            pi_l_old, pi_info_old = ppo_compute_loss_pi(data)
            pi_l_old = pi_l_old.item()

            # Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):

                loss_pi, pi_info = ppo_compute_loss_pi(data)
                # kl = mpi_avg(pi_info['kl'])
                kl = pi_info['kl']
                if kl > 1.5 * target_kl:
                    logger.log('Early stopping at step %d due to reaching max kl.' % i)
                    break

                pi_optimizer.zero_grad()
                loss_pi.backward()
                # mpi_avg_grads(ac.pi)  # average grads across MPI processes
                pi_optimizer.step()

            logger.store(StopIter=i)

            # Log changes from update
            kl, cf = pi_info['kl'], pi_info['cf']
            logger.store(LossPi=pi_l_old, KL=kl, ClipFrac=cf,
                         DeltaLossPi=(loss_pi.item() - pi_l_old))

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

            virtual_ppo()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VirtualEpRet', with_min_and_max=True)
            logger.log_tabular('VirtualEpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('LossModel', average_only=True)
            logger.log_tabular('reg', average_only=True)
            logger.log_tabular('rec', average_only=True)
            logger.log_tabular('loss_d', average_only=True)
            logger.log_tabular('loss_r', average_only=True)
            logger.log_tabular('kl', average_only=True)
            logger.log_tabular('prior_logprob', average_only=True)
            logger.log_tabular('log_det', average_only=True)
            logger.log_tabular('conditional_log_det', average_only=True)
            logger.log_tabular('conditional_logprob', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('ClipFrac', average_only=True)
            logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
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

    print("MBFQ Experiment")
    mbfq(lambda: gym.make(args.env), ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
         gamma=args.gamma, seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs,
         device=args.device, batch_size=args.batch_size)
