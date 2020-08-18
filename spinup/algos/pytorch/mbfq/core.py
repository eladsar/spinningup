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
import torch.autograd as autograd
import itertools
import math
from nflib.flows import AutoEncoderFlow, ConditionalNormalizingFlowModel
from nflib.flows import (
    AffineConstantFlow, ActNorm, AffineHalfFlow,
    SlowMAF, MAF, IAF, Invertible1x1Conv,
    NormalizingFlow, NormalizingFlowModel,
)
from nflib.spline_flows import NSF_AR, NSF_CL
from .spline_model import SplineAutoEncoder, MLP2Layers
from nflib.nets import MLP
import torch.autograd as autograd

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


class WorldModel(nn.Module):

    def __init__(self, obs_dim, state_dim, act_dim):
        super().__init__()

        # self.ae = AutoEncoderFlow(obs_dim, obs_dim // 2, 256, w=1e-3)
        self.ae = SplineAutoEncoder(obs_dim, nh=state_dim)
        self.r = MLP2Layers(state_dim + act_dim, 1, 64)
        self.d = MLP2Layers(state_dim + act_dim, 1, 64)

        flows = [NSF_CL(dim=state_dim, K=60, B=60, hidden_dim=16, base_network=MLP, contex_dim=state_dim+act_dim) for _ in range(3)]
        convs = [Invertible1x1Conv(dim=state_dim) for _ in flows]
        # norms = [ActNorm(dim=state_dim, eps=1e-3) for _ in flows]
        norms = [ActNorm(dim=state_dim) for _ in flows]
        flows = list(itertools.chain(*zip(norms, convs, flows)))

        self.flow = ConditionalNormalizingFlowModel(flows, state_dim, state_dim+act_dim, 128)

    def forward(self, o, a, r, o2, d):

        s, ohat = self.ae(o)
        s2, o2hat = self.ae(o2)

        c = torch.cat([s, a], dim=1)
        rhat = self.r(c)
        dhat = self.d(c)

        _, prior_logprob, log_det = self.flow(s2, c)
        prior_logprob = prior_logprob.mean()
        log_det = log_det.mean()

        s2_nll = -(prior_logprob + log_det)

        loss_r = F.smooth_l1_loss(rhat, r, reduction='sum')
        loss_d = F.binary_cross_entropy_with_logits(dhat, d, reduction='mean')

        reg = torch.square(s).mean(dim=0)
        reg = (reg - torch.log(reg)).sum(dim=-1)

        reg2 = torch.square(s2).mean(dim=0)
        reg2 = (reg2 - torch.log(reg2)).sum(dim=-1)

        rec = torch.square(o - ohat).sum(dim=-1).mean()
        rec2 = torch.square(o2 - o2hat).sum(dim=-1).mean()

        loss = 0.5 * (0.01 * (reg - 1) + rec + 0.01 * (reg2 - 1) + rec2) + loss_r + loss_d + s2_nll

        aux = dict(loss=float(loss), reg=float(reg), reg2=float(reg2), prior_logprob=float(prior_logprob),
                   log_det=float(log_det), loss_d=float(loss_d))

        return loss, aux

    def get_state(self, o, batch_size):

        if len(o) == 1:
            self.ae.eval()

            # try batch correction
            running_mean = self.ae.encoder.embedding.norm.running_mean
            running_var = self.ae.encoder.embedding.norm.running_var

            o0 = o.squeeze(0)

            self.ae.encoder.embedding.norm.running_mean = 1 / batch_size * o0 + (1 - 1 / batch_size) * running_mean
            self.ae.encoder.embedding.norm.running_var = 1 / batch_size * (o0 - running_mean) ** 2 + (1 - 1 / batch_size) * running_var

        with torch.no_grad():
            s, _ = self.ae(o)

        if len(o) == 1:
            self.ae.train()
            self.ae.encoder.embedding.norm.running_mean = running_mean
            self.ae.encoder.embedding.norm.running_var = running_var

        return s

    def gen(self, o, a):

        # only for learning (dont use with a single state)
        with torch.no_grad():
            s, _ = self.ae(o)
            c = torch.cat([s, a], dim=1)
            s2 = self.flow.sample(c)
            r = self.r(c)
            d = torch.bernoulli(torch.sigmoid(self.d(c)))

        return s, s2, r, d

    def grad_q(self, o, a, gamma, v_net):

        with torch.no_grad():
            s, _ = self.ae(o)
            c = torch.cat([s, a], dim=1)
            s2 = self.flow.sample(c)
            v2 = v_net(s2)
            d = torch.bernoulli(torch.sigmoid(self.d(c)))

        a = autograd.Variable(a.detach().clone(), requires_grad=True)
        c = torch.cat([s, a], dim=1)

        rhat = self.r(c)
        _, prior_logprob, log_det = self.flow(s2, c)

        qa = (rhat + gamma * v2 * (1 - d) * (log_det + prior_logprob)).mean()

        grad_q = autograd.grad(outputs=qa, inputs=a, grad_outputs=torch.cuda.FloatTensor(qa.size()).fill_(1.),
                               create_graph=False, retain_graph=False, only_inputs=True)[0]

        return grad_q.detach()


class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = ActorFC(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.v = CriticFC(obs_dim, 0, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic=deterministic, with_logprob=False)
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


def identity(x):
    return x


def zero(x):
    return torch.zeros_like(x)


class ActorFC(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):

        super(ActorFC, self).__init__()

        self.act_limit = act_limit
        self.lin = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.std_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.distribution = None

    def _sample(self, method, n=None, deterministic=False, ratio=1., mean=None):

        self.distribution.logscale += math.log(ratio)

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

        if mean is not None:

            org_mean = self.distribution.loc.expand_as(sample)
            new_mean = atanh(mean.expand_as(sample) / self.act_limit)

            sample = sample - org_mean + new_mean

        a = torch.tanh(sample) * self.act_limit

        self.distribution.logscale -= math.log(ratio)
        return a

    def rsample(self, n=None, deterministic=False, ratio=1., mean=None):
        return self._sample('rsample', n=n, deterministic=deterministic, ratio=ratio, mean=mean)

    def sample(self, n=None, deterministic=False, ratio=1., mean=None):
        return self._sample('sample', n=n, deterministic=deterministic, ratio=ratio, mean=mean)

    def log_prob(self, a):

        distribution = self.distribution.expand(a.shape)

        logp_pi = distribution.log_prob(a).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(axis=-1)

        return logp_pi

    def forward(self, s, deterministic=False, with_logprob=True):

        s = self.lin(s)

        mu = self.mu_head(s)

        logstd = self.std_head(s)
        logstd = torch.clamp(logstd, LOG_STD_MIN, LOG_STD_MAX)

        params = {'loc': mu, 'logscale': logstd}
        self.distribution = MyNormal(**params)

        if deterministic:
            a = mu
        else:
            a = self.distribution.rsample()

        logp_a = self.log_prob(a) if with_logprob else None

        a = torch.tanh(a)
        a = self.act_limit * a

        return a, logp_a


class ReSU(nn.Module):

    def __init__(self):
        super(ReSU, self).__init__()

    def forward(self, x):
        return 0.5 * torch.clamp_min(x, 0) ** 2


class CriticFC(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_size=1):
        super(CriticFC, self).__init__()

        self.actions = act_dim
        self.lin = mlp([obs_dim + act_dim] + list(hidden_sizes) + [output_size], activation)

    def forward_tag(self, s, a, no_grad=False):

        a = autograd.Variable(a.detach().clone(), requires_grad=True)
        geps = self.forward(s, a)

        geps = autograd.grad(outputs=geps, inputs=a, grad_outputs=torch.cuda.FloatTensor(geps.size()).fill_(1.),
                                  create_graph=not no_grad, only_inputs=True)[0]

        return geps

    def forward(self, s, a=None):

        shape = s.shape
        if self.actions:

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