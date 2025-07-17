import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
import numpy as np
import math
from tqdm import tqdm


def get_sde(name, **kwargs):
    if name == 'vpsde':
        return VPSDE(**kwargs)
    elif name == 'vpsde_cosine':
        return VPSDECosine(**kwargs)
    else:
        raise NotImplementedError


def stp(s, ts: torch.Tensor):  
    """ scalar tensor product"""
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):
    """mean of square"""
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)


class SDE(object):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def drift(self, x, t):
        raise NotImplementedError

    def diffusion(self, t):
        raise NotImplementedError

    def cum_beta(self, t):  # the variance of xt|x0
        raise NotImplementedError

    def cum_alpha(self, t):
        raise NotImplementedError

    def snr(self, t):  # signal noise ratio
        raise NotImplementedError

    def nsr(self, t):  # noise signal ratio
        raise NotImplementedError

    def marginal_prob(self, x0, t):  # the mean and std of q(xt|x0)
        alpha = self.cum_alpha(t)
        beta = self.cum_beta(t)
        mean = stp(alpha ** 0.5, x0)  # E[xt|x0]
        std = beta ** 0.5  # Cov[xt|x0] ** 0.5
        return mean, std

    def sample(self, x0, t_init=0):
        """sample from q(xn|x0), where n is uniform. This is from the forward process."""
        t = torch.rand(x0.shape[0], device=x0.device) * (1. - t_init) + t_init
        mean, std = self.marginal_prob(x0, t)
        eps = torch.randn_like(x0)
        xt = mean + stp(std, eps)
        return t, eps, xt


class VPSDE(SDE):
    """VPSDE

    是 Variance Preserving Stochastic Differential Equation（方差保持随机微分方程）的缩写，是扩散模型中常用的一种 SDE。它的主要作用是定义数据的前向扩散过程（即噪声注入过程），使得数据逐步变为高斯噪声，同时保证在整个过程中信号的方差保持在一定范围内。

    在你的代码中，VPSDE 继承自 SDE 基类，主要实现了以下内容：

    初始化参数：beta_min 和 beta_max 控制扩散过程的噪声强度区间，SNR_scale 用于信噪比调整。
    drift：漂移项，定义了数据在扩散过程中的确定性变化。
    diffusion：扩散项，定义了数据在扩散过程中的随机性变化（噪声）。
    cum_alpha / cum_beta：分别表示扩散过程累计的信号和噪声比例（用于计算任意时刻的均值和方差）。
    snr / nsr：信噪比和噪声信号比，衡量当前时刻信号与噪声的比例。
    sample：从任意时刻的扩散分布中采样。
    VPSDE 的核心思想是通过一个随时间变化的 β(t) 控制噪声注入速率，使得数据在 t=0 时为原始数据，t=1 时接近高斯噪声。这样可以方便地进行正向扩散和反向生成。

    """
    def __init__(self, beta_min=0.1, beta_max=20, SNR_scale=1.0):
        # 0 <= t <= 1
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.SNR_scale = SNR_scale
        print(f"using SNR_scale: {self.SNR_scale} for training")

    def drift(self, x, t):
        return -0.5 * stp(self.squared_diffusion(t), x)

    def diffusion(self, t):
        return self.squared_diffusion(t) ** 0.5

    def squared_diffusion(self, t):  # beta(t)
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)

        # SNR adjustment
        a = self.beta_0
        b = self.beta_1 - self.beta_0
        exp_term = torch.exp(-1 * (a * t + 0.5 * b * (t**2)))
        SNR_term_numerator = (self.SNR_scale - 1) * exp_term * (-1 * a - (b * t))
        SNR_term_denominator = 1 + (self.SNR_scale - 1) * exp_term

        return beta_t + (SNR_term_numerator / SNR_term_denominator)

    def squared_diffusion_integral(self, s, t):  # \int_s^t beta(tau) d tau
        integral_beta = self.beta_0 * (t - s) + (self.beta_1 - self.beta_0) * (t ** 2 - s ** 2) * 0.5

        # SNR adjustment
        a = self.beta_0
        b = self.beta_1 - self.beta_0
        exp_term = torch.exp(-1 * (a * (t-s) + 0.5 * b * (t**2 - s**2)))
        SNR_term = torch.log(1 + (self.SNR_scale - 1) * exp_term) - math.log(self.SNR_scale)

        # # verify SNR term
        # intgral_beta_t = self.beta_0 * t + (self.beta_1 - self.beta_0) * (t ** 2) * 0.5
        # cum_alpha = (- intgral_beta_t).exp()  # alpha_bar
        # cum_beta = 1 - cum_alpha
        # org_SNR = cum_alpha / cum_beta
        #
        # exp_term = torch.exp(-1 * (a * t + 0.5 * b * (t ** 2)))
        # SNR_term = torch.log(1 + (self.SNR_scaleup - 1) * exp_term) - math.log(self.SNR_scaleup)
        # intgral_beta_t = intgral_beta_t + SNR_term
        # cum_alpha = (- intgral_beta_t).exp()  # alpha_bar
        # cum_beta = 1 - cum_alpha
        # new_SNR = cum_alpha / cum_beta
        # print(f"SNR scale up by {new_SNR / org_SNR}")

        return integral_beta + SNR_term

    def skip_beta(self, s, t):  # beta_{t|s}, Cov[xt|xs]=beta_{t|s} I
        return 1. - self.skip_alpha(s, t)

    def skip_alpha(self, s, t):  # alpha_{t|s}, E[xt|xs]=alpha_{t|s}**0.5 xs
        x = -self.squared_diffusion_integral(s, t)
        return x.exp()

    def cum_beta(self, t):
        return self.skip_beta(0, t)

    def cum_alpha(self, t):
        return self.skip_alpha(0, t)

    def nsr(self, t):
        return self.squared_diffusion_integral(0, t).expm1()

    def snr(self, t):
        return 1. / self.nsr(t)

    def __str__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'

    def __repr__(self):
        return f'vpsde beta_0={self.beta_0} beta_1={self.beta_1}'


class VPSDECosine(SDE):
    r"""
        dx = f(x, t)dt + g(t) dw with 0 <= t <= 1
        f(x, t) is the drift
        g(t) is the diffusion
    """
    def __init__(self, s=0.008):
        self.s = s
        self.F = lambda t: torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        self.F0 = math.cos(s / (1 + s) * math.pi / 2) ** 2

    def drift(self, x, t):
        ft = - torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi / 2
        return stp(ft, x)

    def diffusion(self, t):
        return (torch.tan((t + self.s) / (1 + self.s) * math.pi / 2) / (1 + self.s) * math.pi) ** 0.5

    def cum_beta(self, t):  # the variance of xt|x0
        return 1 - self.cum_alpha(t)

    def cum_alpha(self, t):
        return self.F(t) / self.F0

    def snr(self, t):  # signal noise ratio
        Ft = self.F(t)
        return Ft / (self.F0 - Ft)

    def nsr(self, t):  # noise signal ratio
        Ft = self.F(t)
        return self.F0 / Ft - 1

    def __str__(self):
        return 'vpsde_cosine'

    def __repr__(self):
        return 'vpsde_cosine'


class ScoreModel(object):
    r"""
        The forward process is q(x_[0,T])
    """

    def __init__(self, nnet: nn.Module, pred: str, sde: SDE, T=1):
        assert T == 1
        self.nnet = nnet
        self.pred = pred
        self.sde = sde
        self.T = T
        print(f'ScoreModel with pred={pred}, sde={sde}, T={T}')

    def predict(self, xt, t, **kwargs):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.to(xt.device)
        if t.dim() == 0:
            t = duplicate(t, xt.size(0))
        print(f"Model input timesteps: {t.shape}, values: {t[:5]}")  # 添加调试信息
        scaled_t = t * 999
        return self.nnet(x=xt, timesteps=scaled_t, **kwargs)  # follow SDE

    def noise_pred(self, xt, t, **kwargs):
        pred = self.predict(xt, t, **kwargs)
        if self.pred == 'noise_pred':
            noise_pred = pred
        elif self.pred == 'x0_pred':
            noise_pred = - stp(self.sde.snr(t).sqrt(), pred) + stp(self.sde.cum_beta(t).rsqrt(), xt)
        else:
            raise NotImplementedError
        return noise_pred

    def x0_pred(self, xt, t, **kwargs):
        pred = self.predict(xt, t, **kwargs)
        if self.pred == 'noise_pred':
            x0_pred = stp(self.sde.cum_alpha(t).rsqrt(), xt) - stp(self.sde.nsr(t).sqrt(), pred)
        elif self.pred == 'x0_pred':
            x0_pred = pred
        else:
            raise NotImplementedError
        return x0_pred

    def score(self, xt, t, **kwargs):
        cum_beta = self.sde.cum_beta(t)
        noise_pred = self.noise_pred(xt, t, **kwargs)
        return stp(-cum_beta.rsqrt(), noise_pred)


class ReverseSDE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt + g(t) dw
    """
    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, t, **kwargs)
        return drift - stp(diffusion ** 2, score)

    def diffusion(self, t):
        return self.sde.diffusion(t)


class ODE(object):
    r"""
        dx = [f(x, t) - g(t)^2 s(x, t)] dt
    """

    def __init__(self, score_model):
        self.sde = score_model.sde  # the forward sde
        self.score_model = score_model

    def drift(self, x, t, **kwargs):
        drift = self.sde.drift(x, t)  # f(x, t)
        diffusion = self.sde.diffusion(t)  # g(t)
        score = self.score_model.score(x, t, **kwargs)
        return drift - 0.5 * stp(diffusion ** 2, score)

    def diffusion(self, t):
        return 0


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


@ torch.no_grad()
def euler_maruyama(rsde, x_init, sample_steps, eps=1e-3, T=1, trace=None, verbose=False, **kwargs):
    r"""
    The Euler Maruyama sampler for reverse SDE / ODE
    See `Score-Based Generative Modeling through Stochastic Differential Equations`
    """
    assert isinstance(rsde, ReverseSDE) or isinstance(rsde, ODE)
    print(f"euler_maruyama with sample_steps={sample_steps}")
    timesteps = np.append(0., np.linspace(eps, T, sample_steps))
    timesteps = torch.tensor(timesteps).to(x_init)
    x = x_init
    if trace is not None:
        trace.append(x)
    for s, t in list(zip(timesteps, timesteps[1:]))[::-1]:
        drift = rsde.drift(x, t, **kwargs)
        diffusion = rsde.diffusion(t)
        dt = s - t
        mean = x + drift * dt
        sigma = diffusion * (-dt).sqrt()
        x = mean + stp(sigma, torch.randn_like(x)) if s != 0 else mean
        if trace is not None:
            trace.append(x)
        statistics = dict(s=s, t=t, sigma=sigma.item())
        logging.debug(dct2str(statistics))
    return x


def LSimple(score_model: ScoreModel, x0, pred='noise_pred', reweight=None, **kwargs):
    """Compute the loss for a simple score model."""
    t, noise, xt = score_model.sde.sample(x0)
    if pred == 'noise_pred':
        noise_pred = score_model.noise_pred(xt, t, **kwargs)
        # 使用 F.mse_loss，设置 reduction='none' 以便后续 reweight
        loss = F.mse_loss(noise_pred, noise, reduction='none')
        loss = loss * reweight  # loss re-weighting
        return loss.flatten(start_dim=1).mean(dim=-1)

    elif pred == 'x0_pred':
        x0_pred = score_model.x0_pred(xt, t, **kwargs)
        loss = F.mse_loss(x0_pred, x0, reduction='none')
        loss = loss * reweight  # loss re-weighting
        return loss.flatten(start_dim=1).mean(dim=-1)

    else:
        raise NotImplementedError(pred)
