import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.distributions.exp_family import ExponentialFamily
from numbers import Number
from torch.distributions import constraints

import math


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        self.distribution = None

    def forward(self, obs, distribution=None, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi, None


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = ActorFC(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        # self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = CriticFC(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = CriticFC(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, _ = self.pi(obs, deterministic=deterministic, with_logprob=False, squash=True)
            return a.cpu().numpy()


class MyNormal(ExponentialFamily):

    arg_constraints = {'loc': constraints.real, 'logscale': constraints.real}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def variance(self):
        return self.scale.pow(2)

    def __init__(self, loc, logscale, validate_args=None):

        self.loc, self.logscale = broadcast_all(loc, logscale)

        if isinstance(loc, Number) and isinstance(logscale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(MyNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MyNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.logscale = self.logscale.expand(batch_shape)
        super(MyNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def scale(self):
        return self.logscale.exp()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -((value - self.loc) ** 2) / (2 * self.variance) - self.logscale - math.log(math.sqrt(2 * math.pi))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        raise NotImplementedError

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)


def log_tanh_grad(x):
    return 2 * (np.log(2) - x - F.softplus(-2 * x))


def atanh(x):
    x = torch.clamp(x, min=-1+1e-5, max=1-1e-5)
    return 0.5 * torch.log((1 + x) / (1 - x))


class Deterministic(torch.distributions.distribution.Distribution):

    def __init__(self, loc):
        super(Deterministic, self).__init__()
        self.x = loc

    def rsample(self, sample_shape=torch.Size([])):

        expanded = sample_shape + self.x.shape

        x = self.x.view(torch.Size([1] * len(sample_shape)) + self.x.shape)
        x = x.expand(expanded)

        return x

    def log_prob(self, value):
        return torch.ones_like(self.x)

    def entropy(self):
        return torch.zeros_like(self.x)


def identity(x):
    return x


def zero(x):
    return torch.zeros_like(x)


class Policy(nn.Module):

    def __init__(self, distribution, bounded=True):
        super(Policy, self).__init__()

        if bounded:
            self.squash = torch.tanh
            self.desquash = atanh
            self.da_du = log_tanh_grad

        else:
            self.squash = identity
            self.desquash = identity
            self.da_du = zero

        if distribution == 'deterministic':
            self.generator = Deterministic
        elif distribution == 'Normal':
            self.generator = MyNormal
        else:
            self.generator = getattr(torch.distributions, distribution)

        self.distribution = None

    def _sample(self, method, n=None, deterministic=False):

        if deterministic:
            sample = self.distribution.mean
            if type(n) is int:
                sample = torch.repeat_interleave(sample.unsqueeze(0), n, dim=0)

            if method == 'sample':
                sample = sample.detach()

        elif n is None:
            sample = getattr(self.distribution, method)()
        else:

            if type(n) is int:
                n = torch.Size([n])
            sample = getattr(self.distribution, method)(n)

        a = self.squash(sample)
        return a

    def rsample(self, n=None, deterministic=False):
        return self._sample('rsample', n=n, deterministic=deterministic)

    def sample(self, n=None, deterministic=False):
        return self._sample('sample', n=n, deterministic=deterministic)

    def log_prob(self, a):

        distribution = self.distribution.expand(a.shape)

        sample = self.desquash(a)
        log_prob = distribution.log_prob(sample) - self.da_du(sample)

        return log_prob.sum(dim=-1)

    def forward(self, deterministic=False, **params):

        self.params = params

        self.distribution = self.generator(**params)
        a = self.rsample(deterministic=deterministic)

        return a.squeeze(0)


class ActorFC(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):

        super(ActorFC, self).__init__()

        self.act_limit = act_limit
        self.lin = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.std_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.distribution = None

    def _sample(self, method, n=None, deterministic=False):

        if deterministic:
            sample = self.distribution.mean

            if type(n) is int:
                sample = torch.repeat_interleave(sample.unsqueeze(0), n, dim=0)

            if method == 'sample':
                sample = sample.detach()

        elif n is None:
            sample = getattr(self.distribution, method)()
        else:

            if type(n) is int:
                n = torch.Size([n])
            sample = getattr(self.distribution, method)(n)

        a = torch.tanh(sample) * self.act_limit
        return a

    def rsample(self, n=None, deterministic=False):
        return self._sample('rsample', n=n, deterministic=deterministic)

    def sample(self, n=None, deterministic=False):
        return self._sample('sample', n=n, deterministic=deterministic)

    def normalize(self, a):

        a = atanh(1 / self.act_limit * a)
        distribution = self.distribution.expand(a.shape)
        a = (a - distribution.loc) / (distribution.scale + 1e-5)
        # a = torch.tanh(a / 10) * self.act_limit * 10
        # a = torch.tanh(a) * self.act_limit
        return a

    def log_prob(self, a, desquash=False):

        a_desquash = atanh(a / self.act_limit) if desquash else a
        distribution = self.distribution.expand(a_desquash.shape)

        logp_pi = distribution.log_prob(a_desquash).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - a_desquash - F.softplus(-2 * a_desquash))).sum(axis=-1)

        return logp_pi

    def forward(self, s, distribution=None, deterministic=False, with_logprob=True, squash=False):

        s = self.lin(s)

        mu = self.mu_head(s)

        logstd = self.std_head(s)
        logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)

        # params = {'loc': mu, 'scale': torch.exp(logstd)}
        # self.distribution = Normal(**params)

        params = {'loc': mu, 'logscale': logstd}
        self.distribution = MyNormal(**params)

        if deterministic:
            a = mu
        else:
            a = self.distribution.rsample()

        logp_a = self.log_prob(a) if with_logprob else None

        if squash:
            a = torch.tanh(a)
            a = self.act_limit * a

        a_norm = None
        if distribution is not None:
            distribution = distribution.expand(a.shape)
            a_norm = (a - distribution.loc) / (distribution.scale + 1e-5)
            # a_norm = torch.tanh(a_norm / 10) * self.act_limit * 10

        return a, logp_a, a_norm


# class ActorFC(Policy):
#
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit,
#                  distribution='Normal', bounded=True):
#
#         super(ActorFC, self).__init__(distribution, bounded=bounded)
#
#         self.lin = mlp([obs_dim] + list(hidden_sizes), activation, activation)
#         self.mu_head = nn.Linear(hidden_sizes[-1], act_dim)
#
#         self.form = distribution
#         if distribution in ['Normal', 'Uniform']:
#             self.std_head = nn.Linear(hidden_sizes[-1], act_dim)
#
#     def forward(self, s, deterministic=False, with_logprob=True):
#
#         s = self.lin(s)
#
#         mu = self.mu_head(s)
#
#         if self.form in ['Normal', 'Uniform']:
#             logstd = self.std_head(s)
#             logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)
#             params = {'loc': mu, 'logscale': logstd}
#         else:
#             params = {'loc': mu}
#
#         a = super(ActorFC, self).forward(**params, deterministic=deterministic)
#         logp_a = self.log_prob(a.detach()) if with_logprob else None
#
#         return a, logp_a


class CriticFC(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(CriticFC, self).__init__()

        self.actions = act_dim
        self.lin = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, s, a=None, distribution=None):

        shape = s.shape
        if self.actions:

            if distribution is not None:
                distribution = distribution.expand(a.shape)
                a = a * (distribution.scale + 1e-5) + distribution.loc

            a = torch.tanh(a)  # * act_limit

            if len(a.shape) > len(shape):
                shape = a.shape
                n, b, _ = shape

                s = s.unsqueeze(0).expand(n, b, -1)

                s = torch.cat([s, a], dim=-1)
                s = s.view(n * b, -1)
            else:
                s = torch.cat([s, a], dim=-1)

        q = self.lin(s)
        q = q.view(*shape[:-1], -1).squeeze(-1)

        return q