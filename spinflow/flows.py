"""
Implements various flows.
Each flow is invertible so it can be forward()ed and backward()ed.
Notice that backward() is not backward as in backprop but simply inversion.
Each flow also outputs its log det J "regularization"

Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from spinflow.nets import LeafParam, MLP, ARMLP
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.distributions.exp_family import ExponentialFamily
from numbers import Number
from torch.distributions import constraints
import itertools
from .spline_flows import NSF_CL
from .spline_embeddings import SplineNet

import math


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
    def kl_normal(self):
        return (self.variance + self.mean ** 2) / 2 - self.logscale - 0.5

    @property
    def variance(self):
        return torch.exp(self.logscale * 2)

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


class AffineHalfEmbeddingFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, dim, parity, scale=True, shift=True, emb=16, layer=64, delta=10, n_res=2):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = SplineNet(self.dim // 2, emb=emb, layer=layer, delta=delta, n=self.dim // 2, n_res=n_res)
        if shift:
            self.t_cond = SplineNet(self.dim // 2, emb=emb, layer=layer, delta=delta, n=self.dim // 2, n_res=n_res)

    def forward(self, x, c=None):
        x0, x1 = x[:, ::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0  # untouched half
        z1 = torch.exp(s) * x1 + t  # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z, c=None):
        z0, z1 = z[:, ::2], z[:, 1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0  # this was the same
        x1 = (z1 - t) * torch.exp(-s)  # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class AutoEncoderFlow(nn.Module):

    def __init__(self, nin, nout, nh, w=1e-3):
        super().__init__()
        self.enc = MLP(nin, nout, nh)
        self.dec = MLP(nout, nin, nh)
        self.w = w

    def forward(self, x, recursion=True):

        z = self.enc(x)

        if recursion:
            x_hat, _ = self.backward(z, recursion=False)
            mse = torch.square(x_hat - x.detach()).sum(dim=1) + self.w * torch.square(z).sum(dim=1)
        else:
            mse = torch.zeros(len(x), device=x.device)

        return z.detach(), -mse

    def backward(self, z, recursion=True):

        x = self.dec(z)

        if recursion:

            z_hat, _ = self.forward(x, recursion=False)
            mse = torch.square(z_hat - z.detach()).sum(dim=1)
        else:
            mse = torch.zeros(len(z), device=z.device)

        return x, -mse


class AffineConstantFlow(nn.Module):
    """ 
    Scales + Shifts the flow by (learned) constants per dimension.
    In NICE paper there is a Scaling layer which is a special case of this where t is None
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        
    def forward(self, x, c=None):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z, c=None):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ActNorm(AffineConstantFlow):
    """
    Really an AffineConstantFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
        self.eps = 1e-3

    def forward(self, x, c=None):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(torch.clamp_min(x.std(dim=0, keepdim=True), min=self.eps))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)


class AffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the 
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)
        
    def forward(self, x, c=None):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z, c=None):
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class SlowMAF(nn.Module):
    """ 
    Masked Autoregressive Flow, slow version with explicit networks per dim
    """
    def __init__(self, dim, parity, net_class=MLP, nh=24):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleDict()
        self.layers[str(0)] = LeafParam(2)
        for i in range(1, dim):
            self.layers[str(i)] = net_class(i, 2, nh)
        self.order = list(range(dim)) if parity else list(range(dim))[::-1]
        
    def forward(self, x, c=None):
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            log_det += s
        return z, log_det

    def backward(self, z, c=None):
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s
        return x, log_det

class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
    def __init__(self, dim, parity, net_class=ARMLP, nh=24):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim*2, nh)
        self.parity = parity

    def forward(self, x, c=None):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z, c=None):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0))
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone()) # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

class IAF(MAF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        reverse the flow, giving an Inverse Autoregressive Flow (IAF) instead, 
        where sampling will be fast but density estimation slow
        """
        self.forward, self.backward = self.backward, self.forward


class Invertible1x1Conv(nn.Module):
    """ 
    As introduced in Glow paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        # remains fixed during optimization
        self.register_buffer('P', P)
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S

        self.W_inv = None

    def _assemble_W(self):
        """ assemble W from its pieces (P, L, U, S) """
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.L.device))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x, c=None):
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        self.W_inv = None
        return z, log_det

    def backward(self, z, c=None):

        if self.W_inv is None:
            W = self._assemble_W()
            self.W_inv = torch.inverse(W)

        x = z @ self.W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det


# ------------------------------------------------------------------------

class NormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = torch.zeros(m, device=x.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m, device=z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def backward(self, z):
        xs, log_det = self.flow.backward(z)
        return xs, log_det

    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs


class ConditionalNormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, c=None):
        m, _ = x.shape
        log_det = torch.zeros(m, device=x.device)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x, c)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z, c=None):
        m, _ = z.shape
        log_det = torch.zeros(m, device=z.device)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, c)
            log_det += ld
            xs.append(z)
        return xs, log_det


class ConditionalNormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, x_dim, c_dim, B=20, K=20, nh=128):
        super().__init__()

        # Define the primery flow

        flows = [NSF_CL(dim=x_dim, K=K, B=B, hidden_dim=16, base_network=MLP) for _ in range(1)]
        convs = [Invertible1x1Conv(dim=x_dim) for _ in flows]
        norms = [ActNorm(dim=x_dim) for _ in flows]
        flows = list(itertools.chain(*zip(norms, convs, flows)))
        self.flow_x = ConditionalNormalizingFlow(flows)

        # Define the secondary flow

        flows = [NSF_CL(dim=x_dim, K=K, B=B, hidden_dim=16, base_network=MLP) for _ in range(1)]
        convs = [Invertible1x1Conv(dim=x_dim) for _ in flows]
        norms = [ActNorm(dim=x_dim) for _ in flows]
        flows = list(itertools.chain(*zip(norms, convs, flows)))
        self.flow_c = ConditionalNormalizingFlow(flows)

        self.x_loc = MLP(c_dim, x_dim, nh, leak=0)
        self.x_logscale = MLP(c_dim, x_dim, nh, leak=0)

        self.register_buffer('prior_loc', torch.zeros(x_dim))
        self.register_buffer('prior_logscale', torch.zeros(x_dim))
        self.conditional_prior = None

    def forward(self, x, c):

        prior = MyNormal(loc=self.prior_loc, logscale=self.prior_logscale)

        loc = self.x_loc(c)
        logscale = self.x_logscale(c)

        params = {'loc': loc, 'logscale': logscale}
        conditional_prior = MyNormal(**params)

        kl = conditional_prior.kl_normal.sum(dim=1)

        self.conditional_prior = conditional_prior

        zs, log_det = self.flow_x.forward(x)
        z = zs[-1]

        prior_logprob = prior.log_prob(z).sum(1)

        zs, log_det_conditional = self.flow_c.forward(z)
        z = zs[-1]

        conditional_log_det = log_det + log_det_conditional
        conditional_logprob = conditional_prior.log_prob(z).sum(1)

        return prior_logprob, conditional_logprob, log_det, conditional_log_det, kl

    def freeze(self, fr=True):
        for param in self.parameters():
            param.requires_grad = not fr

    def sample(self, c=None, num_samples=1):

        if c is not None:
            loc = self.x_loc(c)
            logscale = self.x_logscale(c)

            params = {'loc': loc, 'logscale': logscale}
            conditional_prior = MyNormal(**params)

            z = conditional_prior.sample((num_samples,))
            _, b, _ = z.shape
            z = z.view(num_samples * b, -1)

            zs, _ = self.flow_c.backward(z)
            z = zs[-1]

        else:

            prior = MyNormal(loc=self.prior_loc, logscale=self.prior_logscale)
            z = prior.sample((num_samples,))

        xs, _ = self.flow_x.backward(z)
        x = xs[-1]

        if c is not None:
            x = x.view(num_samples, b, -1)

        return x.squeeze(0)

