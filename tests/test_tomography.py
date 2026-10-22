import numpy as np

from graybox.tomography import angles, identify, kalman, simulate


def test_iv_recovers_dynamics_where_regression_attenuates():
    K = 4
    F = 0.8 * np.eye(K)
    c = np.full(K, 0.3)
    Q = 0.05 * np.eye(K)
    tr = [angles(x) for x in simulate(500, 8, F, c, Q, n=4, seed=1)]
    model = identify(tr)
    assert abs(np.diag(model.F).mean() - 0.8) < 0.08
    Y0 = np.concatenate([Y[:-1] for Y, _ in tr]); Y1 = np.concatenate([Y[1:] for Y, _ in tr])
    naive = ((Y1 - Y1.mean(0)).T @ (Y0 - Y0.mean(0))) @ np.linalg.inv((Y0 - Y0.mean(0)).T @ (Y0 - Y0.mean(0)))
    assert np.diag(naive).mean() < 0.6                    # the noise attenuates plain regression


def test_kalman_forecast_and_calibration():
    K = 3
    F = 0.9 * np.eye(K)
    c = np.full(K, 0.15)
    Q = 0.03 * np.eye(K)
    tr = [angles(x) for x in simulate(400, 10, F, c, Q, n=8, seed=2)]
    model = identify(tr[:300])
    nis = np.concatenate([kalman(Y, R, model)[2] for Y, R in tr[300:]])
    assert 0.6 < nis.mean() / K < 1.6
    fc, fl, _ = kalman(*tr[300], model)
    assert fc.shape == fl.shape == (10, K)


def test_measure_counts_answers():
    from graybox.tomography import FORWARD_QUESTIONS, measure
    ask = lambda msgs, n: ["no"] * n if "comply" in msgs[-1]["content"] else ["yes", "no", "maybe", "yes"][:n]  # noqa: E731
    C = measure(ask, [{"role": "user", "content": "task"}], n=4)
    assert C.shape == (len(FORWARD_QUESTIONS), 2)
    assert list(C[0]) == [0, 4] and list(C[1]) == [2, 3]
