"""Cross-fitted nonparametric estimates of the portion of the hidden adjoint missed by a short head."""
import equinox as eqx
import jax
import jax.numpy as j

from .online_adjoint import teacher_loss


def levels(kind):
    return (1,)


@eqx.filter_jit
def fit(model, data, k, kind):
    # The hash is fixed. Its bin means are learned from the opposing fold of the current batch.
    return jax.random.rademacher(jax.random.key(22517), (model.get_lm_head().axis_size("embed"), 8),
                                 dtype=j.float32).astype(j.bfloat16)


def proxy(x, w, y, k):
    pos = j.sum(x * w[y], axis=-1, dtype=j.float32)
    head = (x @ w[:k].T).astype(j.float32)
    loghead = jax.nn.logsumexp(head, axis=-1)
    logtail = jax.nn.logsumexp(head[..., k//2:], axis=-1) - j.log(2.)
    z = j.logaddexp(j.logaddexp(loghead, logtail), j.where(y >= k, pos, -j.inf))
    return z - pos, j.exp(logtail-z)


def objective(x, w, y, weights, key, config, projection, offsets=None):
    code, n = config
    method, k = divmod(code, 100000)
    b, t, d = x.shape
    width = t // n
    stop = jax.lax.stop_gradient
    def fold(a):
        shape = (b, n//2, 2, width, *a.shape[2:])
        return j.moveaxis(a.reshape(shape), 2, 0).reshape(2, -1, width, *a.shape[2:])

    cheap, mass = proxy(x, w, y, k)
    # Each independent uniform draw labels one position in a contiguous stratum.
    where = (jax.random.randint(key, (2, b*n//2, 1), 0, width)
             if offsets is None else offsets.reshape(2, b*n//2, 1))
    grid = fold(j.arange(b*t).reshape(b,t))
    ids = j.take_along_axis(grid, where, axis=2).reshape(-1)
    h = x.reshape(-1,d)[ids]
    target = y.reshape(-1)[ids]
    full, gold_adjoint = teacher_loss(h, w, target)
    cheap_adjoint = jax.grad(lambda a: proxy(a, w, target, k)[0].sum())(h)
    response = stop((gold_adjoint-cheap_adjoint).astype(j.float32)).reshape(2, -1, d)

    bits = 4 if method in (270,272,274,276) else 6
    count = 2**bits
    direction = w[y] if method in (272,273) else x
    signs = (direction @ projection[:, :bits]) > 0
    bins = (signs.astype(j.int32) * (2**j.arange(bits))).sum(-1)
    if method in (274,275):
        bins = (y % count)
    if method in (276,277):
        bins += count*(mass > .12)
        count *= 2
    bins = fold(stop(bins))
    samples = j.take_along_axis(bins, where, axis=2)[:,:,0]
    def fit_fold(index, value):
        totals = jax.ops.segment_sum(value, index, count)
        counts = jax.ops.segment_sum(j.ones(index.shape), index, count)
        prior = value.mean(axis=0)
        return (totals + 8.*prior) / (counts[:,None] + 8.)
    tables = jax.vmap(fit_fold)(samples, response)
    pred = stop(jax.vmap(lambda table, index: table[index])(tables[::-1], bins))
    xx = fold(x)
    estimate = fold(cheap) + j.sum((xx-stop(xx))*pred.astype(xx.dtype), axis=-1, dtype=j.float32)
    wt = fold(weights)
    select = lambda a: j.take_along_axis(a, where, axis=2)[:,:,0]
    delta = (full.reshape(2,-1)-select(estimate))*select(wt)
    return ((wt*estimate).sum()+width*delta.sum())/weights.sum()
