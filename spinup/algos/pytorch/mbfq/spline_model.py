import torch
from torch import nn
import math
from spinflow.nets import MLP


def init_weights(net, init='ortho'):
    net.param_count = 0
    for module in net.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear,
                               nn.ConvTranspose2d, nn.ConvTranspose1d)):
            if init == 'ortho':
                torch.nn.init.orthogonal_(module.weight)
            elif init == 'N02':
                torch.nn.init.normal_(module.weight, 0, 0.02)
            elif init in ['glorot', 'xavier']:
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                print('Init style not recognized...')

        net.param_count += sum([p.data.nelement() for p in module.parameters()])


class FCNet(nn.Module):

    def __init__(self, actions, layer, n=1, n_res=1):

        super(FCNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(actions, layer, bias=True),
                                SeqResBlock(n_res, layer),
                                nn.ReLU(),
                                nn.Linear(layer, n, bias=True))

        self.n = n

        init_weights(self, init='ortho')

    def forward(self, x):
        x = self.fc(x)
        if self.n == 1:
            x.squeeze_(1)

        return x


class RunningNorm(nn.Module):

    def __init__(self, shape, momentum=0.01, eps=1e-5):
        super(RunningNorm, self).__init__()

        if type(shape) is int:
            shape = (shape, )

        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(1, *shape))
        self.register_buffer('running_var', torch.ones(1, *shape))
        self.weight = nn.Parameter(torch.Tensor(1, *shape))
        self.bias = nn.Parameter(torch.Tensor(1, *shape))
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):

        if self.training:

            with torch.no_grad():
                mean = x.mean(dim=0, keepdim=True)
                var = x.var(dim=0, keepdim=True)
                self.running_mean = self.running_mean * (1 - self.momentum) + self.momentum * mean
                self.running_var = self.running_var * (1 - self.momentum) + self.momentum * var

        mean = self.running_mean

        std = torch.sqrt(self.running_var) + self.eps

        return (x - mean) / std * self.weight + self.bias


class SplineEmbedding(nn.Module):

    def __init__(self, actions, emb, delta):

        super(SplineEmbedding, self).__init__()

        self.delta = delta
        self.actions = actions
        self.emb = emb

        self.register_buffer('ind_offset', torch.arange(self.actions, dtype=torch.int64).unsqueeze(0))
        self.b = nn.Embedding((2 * self.delta + 1) * actions, emb, sparse=True)

        self.norm = RunningNorm(self.actions, eps=1e-05, momentum=0.1)
        # self.norm = nn.BatchNorm1d(self.actions, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):

        n = len(x)

        x = self.norm(x)
        x = torch.tanh(x)
        x = torch.clamp(x, min=-1+1e-5, max=1-1e-5)

        xl = (x * self.delta).floor()
        xli = self.actions * (xl.long() + self.delta) + self.ind_offset
        xl = xl / self.delta
        xli = xli.view(-1)

        xh = (x * self.delta + 1).floor()
        xhi = self.actions * (xh.long() + self.delta) + self.ind_offset
        xh = xh / self.delta
        xhi = xhi.view(-1)

        bl = self.b(xli).view(n, self.actions, self.emb)
        bh = self.b(xhi).view(n, self.actions, self.emb)

        delta = 1 / self.delta

        x = x.unsqueeze(2)
        xl = xl.unsqueeze(2)
        xh = xh.unsqueeze(2)

        h = bh / delta * (x - xl) + bl / delta * (xh - x)

        return h


class SeqResBlock(nn.Module):

    def __init__(self, n_res, layer):

        super(SeqResBlock, self).__init__()

        self.seq_res = nn.ModuleList([ResBlock(layer) for i in range(n_res)])

    def forward(self, x):

        for res in self.seq_res:
            x = res(x)

        return x


class ResBlock(nn.Module):

    def __init__(self, layer):

        super(ResBlock, self).__init__()

        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Linear(layer, layer, bias=True),
                                nn.ReLU(),
                                nn.Linear(layer, layer, bias=True),
                                )

    def forward(self, x):

        h = self.fc(x)
        return x + h


class GlobalBlock(nn.Module):

    def __init__(self, actions, emb, planes):
        super(GlobalBlock, self).__init__()

        self.actions = actions
        self.emb = emb

        self.query = nn.Sequential(
            #             nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True)
            #             spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
        )

        self.key = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True)
        )

        self.value = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True)
        )

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True)
        )

        self.planes = planes

    def forward(self, x):

        q = self.query(x).transpose(1, 2)
        k = self.key(x)
        v = self.value(x).transpose(1, 2)

        a = torch.softmax(torch.bmm(q, k) / math.sqrt(self.planes), dim=2)
        r = torch.bmm(a, v).transpose(1, 2)
        r = self.output(r)

        return x + r


class GlobalModule(nn.Module):

    def __init__(self, actions, emb, planes):
        super(GlobalModule, self).__init__()

        self.actions = actions
        self.emb = emb

        self.blocks = nn.Sequential(
            GlobalBlock(actions, emb, planes),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):

        x = self.blocks(x)

        return x.squeeze(2)


class SplineHead(nn.Module):

    def __init__(self, actions, emb, emb2, layer, n=1, n_res=1):

        super(SplineHead, self).__init__()

        self.emb = emb
        self.actions = actions
        self.n = n
        self.emb2 = emb2

        self.global_interaction = GlobalModule(actions, emb, emb)

        input_len = emb + actions

        self.fc = nn.Sequential(nn.Linear(input_len, layer, bias=True),
                                SeqResBlock(n_res, layer),
                                nn.ReLU(),
                                nn.Linear(layer, n, bias=True))

        init_weights(self, init='ortho')

    def forward(self, x, x_emb):

        h = x_emb.transpose(2, 1)
        h = self.global_interaction(h)

        x = torch.cat([x, h], dim=1)

        x = self.fc(x)
        x = x.squeeze(1)

        return x


class SplineNet(nn.Module):

    def __init__(self, actions, emb=8, layer=64, delta=10, n=1, n_res=1):

        super(SplineNet, self).__init__()

        emb2 = int(layer / actions + .5)
        self.embedding = SplineEmbedding(actions, emb, delta)
        self.head = SplineHead(actions, emb, emb2, layer, n=n, n_res=n_res)

    def forward(self, x):

        x_emb = self.embedding(x)
        x = self.head(x, x_emb)

        return x


class MLP2Layers(nn.Module):
    """ a simple 2-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SplineAutoEncoder(nn.Module):

    def __init__(self, actions, emb=8, layer=64, delta=10, nh=128, n_res=1):
        super().__init__()

        self.encoder = SplineNet(actions, emb=emb, layer=layer, delta=delta, n=nh, n_res=n_res)
        self.decoder = MLP(nin=nh, nout=actions, nh=2 * layer, leak=0)

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)

        return z, xhat


class SparseDenseAdamOptimizer:
    def __init__(self, nets, dense_args=dict(), sparse_args=dict()):

        if type(nets) is not list:
            nets = [nets]

        sparse_parameters = []
        dense_parameters = []

        for n in nets:
            for m in n.modules():
                for p in m.parameters():
                    if issubclass(type(m), nn.Embedding):
                        sparse_parameters.append(p)
                    dense_parameters.append(p)

        sparse_parameters = list(set(sparse_parameters))
        dense_parameters = list(set(dense_parameters).difference(set(sparse_parameters)))
        opt_sparse = torch.optim.SparseAdam(sparse_parameters, **sparse_args)
        opt_dense = torch.optim.Adam(dense_parameters, **dense_args)

        self.optimizers = (opt_sparse, opt_dense)

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()