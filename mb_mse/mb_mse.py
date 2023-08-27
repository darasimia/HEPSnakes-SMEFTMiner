from functools import partial
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import jax.scipy.stats as stats
from jax.scipy.special import xlogy
from scipy.optimize import minimize

import jax.numpy as jnp
import jax
from jax import config
import jax.tree_util
config.update("jax_debug_nans", True)

import flax.linen as nn
import optax

import numpy as np

import os


θsm = jnp.array([0.0, 0.0])
θ1 = jnp.array([1.0, 0.5])

θspace = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(0, 2, 41),
        jnp.linspace(0, 2, 41),
    ),
    axis=-1,
)

radius, angle = jnp.linspace(0., 1.5, 41), jnp.linspace(0, 0.5*jnp.pi, 41)
θscan = jnp.stack(
    [
        radius * jnp.cos(angle),
        radius * jnp.sin(angle),
    ],
    axis=-1,
)

@partial(jnp.vectorize, signature="(n),(m),()->()")
def p_z_θ_log(z, θ, η):
    p0 = stats.multivariate_normal.logpdf(
        x=z,
        mean=jnp.array([0,0]),
        cov=jnp.diag(jnp.array([1+η, 1+η])),
    )
    p1 = stats.multivariate_normal.logpdf(
        x=z,
        mean=jnp.array([0.5,0.5]),
        cov=jnp.diag(jnp.array([4+η, 0.5+η])),
    )
    p2 = stats.multivariate_normal.logpdf(
        x=z,
        mean=jnp.array([1,1]),
        cov=jnp.diag(jnp.array([0.5+η, 4+η])),
    )
    ptot = (1 + θ.sum())
    return jnp.log((jnp.exp(p0) + θ[0]*jnp.exp(p1) + θ[1]*jnp.exp(p2)) / ptot)

@partial(jnp.vectorize, signature="(k,n),(m),(m),()->(k)")
def llr_z(z, θ1, θ0, η):
    return p_z_θ_log(z, θ1, η) - p_z_θ_log(z, θ0, η)

def templates(rng, n, binning, η):
    keys = jax.random.split(rng, 3)
    s0 = jax.random.multivariate_normal(
        keys[0],
        mean=jnp.array([0,0]),
        cov=jnp.diag(jnp.array([1+η, 1+η])),
        shape=(n,),
    )
    s0 = jnp.histogram2d(s0[:, 0], s0[:, 1], bins=binning)[0] / n
    s1 = jax.random.multivariate_normal(
        keys[1],
        mean=jnp.array([0.5,0.5]),
        cov=jnp.diag(jnp.array([4+η, 0.5+η])),
        shape=(n,),
    )
    s1 = jnp.histogram2d(s1[:, 0], s1[:, 1], bins=binning)[0] / n
    s2 = jax.random.multivariate_normal(
        keys[2],
        mean=jnp.array([1,1]),
        cov=jnp.diag(jnp.array([0.5+η, 4+η])),
        shape=(n,),
    )
    s2 = jnp.histogram2d(s2[:, 0], s2[:, 1], bins=binning)[0] / n
    return s0, s1, s2


def sample(rng, θ, n, η=0.1):
    keys = jax.random.split(rng, 4)
    s0 = jax.random.multivariate_normal(
        keys[0],
        mean=jnp.array([0,0]),
        cov=jnp.diag(jnp.array([1+η, 1+η])),
        shape=(n,),
    )
    s1 = jax.random.multivariate_normal(
        keys[1],
        mean=jnp.array([0.5,0.5]),
        cov=jnp.diag(jnp.array([4+η, 0.5+η])),
        shape=(n,),
    )
    s2 = jax.random.multivariate_normal(
        keys[2],
        mean=jnp.array([1,1]),
        cov=jnp.diag(jnp.array([0.5+η, 4+η])),
        shape=(n,),
    )
    s = jnp.stack([s0, s1, s2])
    idx = jnp.searchsorted(
        jnp.array([0, 1, 1+θ[0], 1+θ.sum()]),
        jax.random.uniform(keys[3], shape=(n,), maxval=1 + θ.sum()),
        side="right",
    )
    return s[idx - 1, jnp.arange(n)]


def templates_lr(rng, n, θ1, θ0):
    keys = jax.random.split(rng, 4)
    s0 = jax.random.multivariate_normal(
        keys[0],
        mean=jnp.zeros(2),
        cov=jnp.eye(2),
        shape=(n,),
    )
    llr0 = llr_z(s0, θ1, θ0, 0)
    s = sample(keys[3], θ1, 1_000_000)
    binning = jnp.linspace(s.min(), s.max(), 101)
    s0 = jnp.histogram(llr0, bins=binning)[0] / n
    s1 = jax.random.multivariate_normal(
        keys[1],
        mean=jnp.zeros(2),
        cov=jnp.diag(jnp.array([4, 0.5])),
        shape=(n,),
    )
    s1 = jnp.histogram(llr_z(s1, θ1, θ0, 0), bins=binning)[0] / n
    s2 = jax.random.multivariate_normal(
        keys[2],
        mean=jnp.zeros(2),
        cov=jnp.diag(jnp.array([0.5, 4])),
        shape=(n,),
    )
    s2 = jnp.histogram(llr_z(s2, θ1, θ0, 0), bins=binning)[0] / n
    return (s0, s1, s2), binning

binning = (
    jnp.linspace(-5, 5, 4),
    jnp.linspace(-5, 5, 4),
)
bname = f"binned {len(binning[0])-1}x{len(binning[1])-1}"
print(bname)

t = templates(jax.random.PRNGKey(29), 10_000_000, binning, 0.1)
print(min(b.min() for b in t))

@partial(jnp.vectorize, signature="(n,m),(l)->()")
def binnedp(zbin, θ):
    ptot = (1 + θ.sum())
    psum = (t[0] + θ[0]*t[1] + θ[1]*t[2]) / ptot
    return xlogy(zbin, psum).sum()

class mb_mse_MLP(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=100)(x)
    x = nn.tanh(x)
    x = nn.Dense(features=100)(x)
    x = nn.tanh(x)
    x = nn.Dense(features=2)(x)
    return x

mb_mse_mlp = mb_mse_MLP()


def log_r(x, z, θnum, θden):
    return p_z_θ_log(z, θnum, 0.1) - p_z_θ_log(z, θden, 0.1)

@jax.jit
def mb_log_rhat(x, θnum, θden, params):
    out = jnp.exp(mb_mse_mlp.apply(params, x))
    return jnp.log((1 + out[:, 0] * θnum[0] + out[:, 1] * θnum[1]) / (1 + θnum[0] + θnum[1])) - jnp.log(1e14)


rng = jax.random.PRNGKey(42)
N = 330000
rng_a, rng_b = jax.random.split(rng, 2)
zsample = sample(rng_a, θsm, N, 0.1)
xsample = zsample + jax.random.multivariate_normal(rng_b, mean=jnp.zeros(2), cov=0.1*jnp.eye(2), shape=(N,))

def make_xsample(rng, theta, N):
    rng_z, rng_x = jax.random.split(rng, 2)
    zsample = sample(rng_z, theta, N, 0.1)
    smear = jax.random.multivariate_normal(
        rng_x,
        mean=jnp.zeros(2),
        cov=0.1*jnp.eye(2),
        shape=(N,),
    )
    return zsample, smear

def make_mse_loss(params, N):
    def mse_loss(params, rng):
        rngs = jax.random.split(rng, 3)
        thetarand = jnp.abs(jax.random.multivariate_normal(rngs[0], mean=jnp.zeros(2), cov=jnp.eye(2)*2))
        samp = make_xsample(rngs[1], θsm, N)
        zs = samp[0]
        xs = zs + samp[1]
        samp_num = make_xsample(rngs[2], thetarand, N)
        zs_num = samp_num[0]
        xs_num = samp_num[1] + zs_num
        #don't divide in log space
        return 1/N * jnp.sum(
        (log_r(xs, zs, thetarand, θsm) - 
         mb_log_rhat(xs, thetarand, θsm, params)
        )**2 + ( ((log_r(xs_num, zs_num, thetarand, θsm))) - 
        (mb_log_rhat(xs_num, thetarand, θsm, params)) )**2
        )
    
    
    return jax.jit(mse_loss)






rng = jax.random.PRNGKey(42)
rng_a, rng_b = jax.random.split(rng, 2)
zsample = sample(rng_a, θsm, 1000, 0.1)
xsample = zsample + jax.random.multivariate_normal(rng_b, mean=jnp.zeros(2), cov=0.1*jnp.eye(2), shape=(1000,))
features = xsample
params = mb_mse_mlp.init(rng, features)
loss = make_mse_loss(params, N)
value_and_grad_fn = jax.value_and_grad(loss)
value_and_grad_fn = jax.jit(value_and_grad_fn)
lr = 1e-3
epochs = 35_000
opt_adam = optax.adam(learning_rate = lr)
opt_state = opt_adam.init(params)
rng = jax.random.PRNGKey(42)
losshist = []

for epoch in range(epochs):
    loss, grads = value_and_grad_fn(params,rng)
    updates, opt_state = opt_adam.update(grads, opt_state)
    params =  optax.apply_updates(params, updates)
    losshist.append(loss)



def gen_2d(nsamps):
    samp = sample(jax.random.PRNGKey(nsamps-1), θ1, n=nsamps)
    sampold = samp
    samp = samp + jax.random.multivariate_normal(
        jax.random.PRNGKey(nsamps),
        mean=jnp.zeros(2),
        cov=jnp.diag(jnp.array([0.1, 0.1])),
        shape=(nsamps,),
    )

    def nll_scan(theta):
        return -mb_log_rhat(samp, theta, θsm, params).sum() 

    out = jax.lax.map(nll_scan, θspace.reshape(-1, 2)).reshape(41,41)
    out = (out - out.min())

    @partial(jnp.vectorize, signature="(n)->()")
    def target_nll_scan(theta):
        return -p_z_θ_log(samp, theta, 0.1).sum()

    target = target_nll_scan(θspace)
    target = target - target.min()

    fig, ax = plt.subplots(figsize=(4,4))

    csb = ax.contour(
        θspace[..., 0], θspace[..., 1],
        out,
        levels=jnp.arange(1, 5)**2/2,
        linestyles="dashed",
        linewidths=0.5,
    )
    cs = ax.contour(
        θspace[..., 0], θspace[..., 1],
        target,
        levels=jnp.arange(1, 5)**2/2,
    )
    ax.clabel(cs, cs.levels, inline=True, fmt=lambda x: "{}σ".format((2*x)**.5), fontsize=10)
    ax.scatter(*θ1, marker="s", label=r"True $\theta$")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.legend()

    plt.savefig(f"gaussian_mixture_prof_samp{len(samp)}_nll_{bname}.pdf", bbox_inches="tight")

gen_2d(50)
gen_2d(1000)
gen_2d(5000)
