import numpy as np
import matplotlib.pyplot as plt
import parameter
import fitting
import evaluate


# r = 0.5 + 0.2sin(5t)
get_r = lambda theta: 0.5 + 0.2 * np.sin(5 * theta)

# dx/dt = cos(5t)cos(t) - r * sin(t)
get_dx_div_dt = lambda theta: np.cos(5 * theta) * np.cos(theta) - np.sin(theta) * get_r(theta)

# dy/dt = cos(5t)sin(t) + r * cos(t)
get_dy_div_dt = lambda theta: np.cos(5 * theta) * np.sin(theta) + np.cos(theta) * get_r(theta)


def get_items(data, begin, end):
    assert begin < end
    m = len(data)
    if begin < 0:
        k = int(np.ceil(-begin * 1.0 / m)) * m
        begin += k
        end += k
    idx = [i % m for i in range(begin, end)]
    return data[idx]


def get_interface(theta):
    '''
    phi(x) = r - (0.5 + 0.2 sin(5 * theta))

    Returns
    -------
    {(x, y) | phi(x) = 0|
    '''
    r = get_r(theta)
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    return np.stack([x, y], axis=1)


def get_dy_div_dx(theta):
    dx_div_dt = get_dx_div_dt(theta)
    dy_div_dt = get_dy_div_dt(theta)
    # dy/dx = (cos(5t)sin(t) + (0.5 + 0.2sin(5t)) * cos(t)) / (cos(5t)cos(t) - (0.5 + 0.2sin(5t)) * sin(t))
    dy_div_dx = dy_div_dt / dx_div_dt
    return dy_div_dx


def get_d2y_div_dx2(theta):
    '''
    y = y(t), x = x(t)

    (d ((dy/dt) / (dx/dt)) / dt) / (dx / dt)
    '''
    r = get_r(theta)
    dx_div_dt = get_dx_div_dt(theta)

    # d2x_div_dt2 = (-5.2 * np.sin(5 * theta) - 0.5) * np.cos(theta) - 2 * np.cos(5 * theta) * np.sin(theta)
    # d2y_div_dt2 = (-5.2 * np.sin(5 * theta) - 0.5) * np.sin(theta) + 2 * np.cos(5 * theta) * np.cos(theta)

    up = (1.04 * np.square(np.sin(5 * theta)) + 2.7 * np.sin(5 * theta) + 2 * np.square(np.cos(5 * theta)) + 0.25)
    down = np.square(np.sin(theta) * (-0.2 * np.sin(5 * theta) - 0.5) + np.cos(theta) * np.cos(5 * theta))
    d2y_div_dx2 = up / down / dx_div_dt

    return d2y_div_dx2


def get_dis(xy):
    diff_xy = xy - np.concatenate([xy[1:, :], xy[0:1, :]])
    dis = np.sqrt(np.square(diff_xy).sum(axis=1))
    return dis


n = 10000
delta = 2 * np.pi / n
theta = np.arange(n) * delta

xy = get_interface(theta)
grad = get_dy_div_dx(theta)
grad2 = get_d2y_div_dx2(theta)

CLIP_UP = 10
CLIP_DOWN = -CLIP_UP

clip_grad = np.clip(grad, CLIP_DOWN, CLIP_UP)
clip_grad2 = np.clip(grad2, CLIP_DOWN, CLIP_UP)
degrees = np.arange(n) * 360 / n

# curvature
K = np.abs(grad2) / np.power(1 + np.square(grad), 1.5)

# chord len
chord_len = get_dis(xy)
cum_chord_len = np.cumsum(chord_len)


def get_uniform_sample(n, xy, standard):
    assert len(xy) == len(standard), (len(xy), len(standard))
    m = len(xy)
    assert m >= 3
    result = np.empty((n, ) + xy.shape[1:])
    result[0, :] = xy[0, :]
    sum_standard = standard.sum() - standard[-1]
    if n > 1:
        interval = sum_standard / (n-1)
        i = 1
        j = 1
        value = standard[0] - interval
        while i < n and j < m:
            if value >= 0:
                result[i, :] = xy[j, :]
                value -= interval
                i += 1
            value += standard[j]
            j += 1
        if i == n - 1 and j >= m:
            result[i, :] = xy[-1, :]
            i += 1
        assert i == n, (i, n, j, m, value)
    return result


def compute_range(min_angle, max_angle):
    min_i = int(np.round(min_angle / delta))
    max_i = int(np.round(max_angle / delta))

    sample_size = 5
    sample_range_size = max_i - min_i + 1
    test_size = sample_range_size
    valid_xy = get_items(xy, min_i, max_i + 1)
    valid_chord_len = get_items(chord_len, min_i, max_i + 1)
    valid_grad = get_items(grad, min_i, max_i + 1)
    valid_grad2 = get_items(grad2, min_i, max_i + 1)
    # sample_xy = get_uniform_sample(sample_size, valid_xy, valid_chord_len)

    sample_xy = get_uniform_sample(sample_size, valid_xy, np.ones(sample_range_size))
    sample_t = parameter.get_uniform(sample_xy)

    test_xy = get_uniform_sample(test_size, valid_xy, np.ones(sample_range_size))
    test_t = parameter.get_uniform(test_xy)
    '''
    bc_type = ([(1, valid_grad[0]), (2, valid_grad2[0])],
            [(1, valid_grad[-1]), (2, valid_grad2[-1])])
    '''
    bc_type = None
    func = fitting.get_bspline(sample_t, sample_xy, k=2, bc_type=bc_type)

    sample_error = evaluate.evaluate(func, sample_t, sample_xy)
    test_error = evaluate.evaluate(func, test_t, test_xy)

    print(sample_error, test_error)

    pred = func(sample_t)
    pred_x = pred[:, 0]
    pred_y = pred[:, 1]
    # print(len(pred_x), sample_error, valid_error)
    plt.plot(sample_xy[:, 0], sample_xy[:, 1], 'k*')
    return pred_x, pred_y


# plt.subplot(121)
# plt.title('x-y')
plt.plot(xy[:, 0], xy[:, 1])

min_angle = np.pi / 10
max_angle = 3 * np.pi / 10

# plt.subplot(122)
# plt.title('pred')
for _ in range(10):
    pred_x, pred_y = compute_range(min_angle, max_angle)
    plt.plot(pred_x, pred_y, 'r')
    plt.axis('equal')
    min_angle += np.pi / 5
    max_angle += np.pi / 5

plt.axis((-1, 1, -1, 1))
plt.show()

plt.subplot(231, projection='polar')
plt.title('polar')
plt.plot(theta, get_r(theta))

plt.subplot(232, projection='polar')
plt.title('dy/dx (polar)')
plt.plot(theta, clip_grad)

plt.subplot(233, projection='polar')
plt.title('d2y/dx2 (polar)')
plt.plot(theta, clip_grad2)

plt.subplot(234)
plt.title('x-y')
plt.plot(xy[:, 0], xy[:, 1])
plt.plot(sample_xy[:, 0], sample_xy[:, 1], 'k*')
plt.plot(valid_xy[:, 0], pred_y, 'r')
plt.axis('equal')
plt.axis((-1, 1, -1, 1))

plt.subplot(235)
plt.title('dy/dx')
plt.plot(degrees, clip_grad)

plt.subplot(236)
plt.title('d2y/dx2')
plt.plot(degrees, clip_grad2)


plt.show()

plt.subplot(321, projection='polar')
plt.title('polar')
plt.plot(theta, get_r(theta))

plt.subplot(322)
plt.title('curvature')
plt.plot(degrees, K)

plt.subplot(323, projection='polar')
plt.title('curvature (polar)')
plt.plot(theta, K)

plt.subplot(324, projection='polar')
plt.title('chord length')
plt.plot(theta, chord_len)

plt.subplot(325)
plt.title('chord length')
plt.plot(degrees, chord_len)

plt.subplot(326)
plt.title('cumulative sum of chord length')
plt.plot(degrees, cum_chord_len)


plt.show()
