import numpy as np
import time

rng = np.random.default_rng()


def test_single(mu, var, N, horizon, shape):
    candidate_action_sequences1 = np.empty(shape)
    for h in range(horizon):
        candidate_action_sequences1[:, h, :] = rng.multivariate_normal(mu[h], np.diag(var[h]), size=N, method='cholesky')


def test_whole(mu, var, N, horizon, shape):
    candidate_action_sequences1 = rng.multivariate_normal(mu, np.diag(var), size=N, method='cholesky')
    candidate_action_sequences2 = candidate_action_sequences1.reshape((N, horizon, -1))
    assert candidate_action_sequences2.shape == shape

def test_cov(mu, cov, N, horizon, shape):
    candidate_action_sequences1 = rng.multivariate_normal(mu, cov, size=N, method='cholesky')
    candidate_action_sequences2 = candidate_action_sequences1.reshape((N, horizon, -1))
    assert candidate_action_sequences2.shape == shape

if __name__ == '__main__':
    N = 1000
    horizon = 30
    ac_dim = 6

    a_elites = rng.uniform(-1, 1, (N, horizon, ac_dim))
    mu = np.mean(a_elites, axis=0)
    var = np.var(a_elites, axis=0)
    times0 = []
    for i in range(50):
        t0 = time.time()
        test_single(mu, var, N, horizon, a_elites.shape)
        times0.append(time.time() - t0)

    times1 = []
    muf = mu.flatten()
    varf = var.flatten()
    cov = np.diag(varf)
    for i in range(50):
        t0 = time.time()
        test_cov(muf,cov, N, horizon, a_elites.shape)
        times1.append(time.time() - t0)

    print(np.mean(times0), np.std(times0))
    print(np.mean(times1), np.std(times1))
