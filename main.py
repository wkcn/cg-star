import numpy as np
import matplotlib.pyplot as plt
import parameter
import fitting
import evaluate


# r = 0.5 + 0.2sin(5t)
def get_r(theta): return 0.5 + 0.2 * np.sin(5 * theta)

# dx/dt = cos(5t)cos(t) - r * sin(t)
def get_dx_div_dt(theta): return np.cos(5 * theta) * \
    np.cos(theta) - np.sin(theta) * get_r(theta)

# dy/dt = cos(5t)sin(t) + r * cos(t)
def get_dy_div_dt(theta): return np.cos(5 * theta) * \
    np.sin(theta) + np.cos(theta) * get_r(theta)


def get_d2x_div_dt2(theta): return (-5.2 * np.sin(5 * theta) - 0.5) * \
    np.cos(theta) - 2 * np.cos(5 * theta) * np.sin(theta)


def get_d2y_div_dt2(theta): return (-5.2 * np.sin(5 * theta) - 0.5) * \
    np.sin(theta) + 2 * np.cos(5 * theta) * np.cos(theta)


def get_items(data, begin, end):
    assert begin < end
    m = len(data)
    if begin < 0:
        k = int(np.ceil(-begin * 1.0 / m)) * m
        begin += k
        end += k
    idx = [i % m for i in range(begin, end)]
    return data[idx]


def get_angle_from_xy(xy):
    with np.errstate(divide='ignore'):
        angle = np.arctan(xy[:, 1] / xy[:, 0])
    left_x = xy[:, 0] < 0
    return angle + left_x * np.pi


assert np.allclose(get_angle_from_xy(np.array([[1, 1], [0, 2], [-1, 0], [-1, -1]])), np.array([np.pi / 4, np.pi / 2, np.pi, np.pi / 4 * 5]))


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

    up = (1.04 * np.square(np.sin(5 * theta)) + 2.7
          * np.sin(5 * theta) + 2 * np.square(np.cos(5 * theta)) + 0.25)
    down = np.square(np.sin(theta) * (-0.2 * np.sin(5 * theta)
                                      - 0.5) + np.cos(theta) * np.cos(5 * theta))
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
curvature = np.abs(grad2) / np.power(1 + np.square(grad), 1.5)

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
        interval = sum_standard / (n - 1)
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


def is_same_end_points(a, b):
    return (a[0] == b[0]).all() and (a[-1] == b[-1]).all()


def compute_range(min_angle, max_angle):
    min_i = int(np.round(min_angle / delta))
    max_i = int(np.round(max_angle / delta))
    print("Range: ", min_i, max_i)

    '''basic'''
    sample_size = 4
    test_size = 10000
    K = 3
    assert 0 <= K < sample_size
    sample_range_size = max_i - min_i

    '''fitting'''
    # fitting_method = fitting.get_bspline
    # fitting_method = fitting.get_bezier
    fitting_method = fitting.get_poly

    '''parameter'''
    parameter_method = parameter.get_uniform
    # parameter_method = parameter.get_cum_chord
    # parameter_method = parameter.get_entad

    '''gradient'''
    USE_GRAD = 0
    assert 0 <= USE_GRAD <= 2

    # Let us denote the points in range [min_i, max_i] valid
    valid_xy = get_items(xy, min_i, max_i)
    valid_chord_len = get_items(chord_len, min_i, max_i)
    valid_curvature = get_items(curvature, min_i, max_i)
    valid_grad = get_items(grad, min_i, max_i)
    valid_grad2 = get_items(grad2, min_i, max_i)
    print("chord_len: ", valid_chord_len.min(), valid_chord_len.max())
    print("curvature: ", curvature.min(), curvature.max())

    '''sampling'''
    sample_xy = get_uniform_sample(sample_size, valid_xy,\
        np.ones(sample_range_size))
        # 1.0 / (1.0 + valid_chord_len / 0.1))
        # 1.0 / (1.0 + valid_curvature/50))

    assert is_same_end_points(sample_xy, valid_xy)

    # parameter for t
    sample_t = parameter_method(sample_xy)

    if USE_GRAD == 0:
        bc_type = None
    else:
        s_min_angle = get_angle_from_xy(sample_xy[0:1])
        s_max_angle = get_angle_from_xy(sample_xy[-1:])
        bc_type = [[], []]
        grad_left = np.array([get_dx_div_dt(s_min_angle), get_dy_div_dt(s_min_angle)])
        grad_right = np.array([get_dx_div_dt(s_max_angle), get_dy_div_dt(s_max_angle)])
        bc_type[0].append((1, grad_left))
        bc_type[1].append((1, grad_right))
        if USE_GRAD == 2:
            grad2_left = np.array([get_d2x_div_dt2(s_min_angle), get_d2y_div_dt2(s_min_angle)])
            grad2_right = np.array([get_d2x_div_dt2(s_max_angle), get_d2y_div_dt2(s_max_angle)])
            bc_type[0].append((2, grad2_left))
            bc_type[1].append((2, grad2_right))


    func = fitting_method(sample_t, sample_xy, k=K, bc_type=bc_type)

    sample_error = evaluate.evaluate(func, sample_t, sample_xy)

    # test
    test_t = np.arange(test_size) / test_size
    pred_xy = func(test_t)
    gt_theta = get_angle_from_xy(pred_xy)
    gt_xy = get_interface(gt_theta)
    test_error = evaluate.diff(pred_xy, gt_xy, order=2)

    # assert is_same_end_points(pred, valid_xy)
    # print(len(pred_x), sample_error, valid_error)
    plt.plot(sample_xy[:, 0], sample_xy[:, 1], 'k*')
    return pred_xy, sample_error, test_error


# plt.subplot(121)
# plt.title('x-y')
plt.plot(xy[:, 0], xy[:, 1])

min_angle = np.pi / 10
max_angle = 3 * np.pi / 10

# plt.subplot(122)
# plt.title('pred')
preds = []
sample_error_sum = 0.0
test_error_sum = 0.0
count = 0
for _ in range(10):
    pred, sample_error, test_error = compute_range(min_angle, max_angle)
    plt.plot(pred[:, 0], pred[:, 1], 'r')
    # preds.append(pred)
    min_angle += np.pi / 5
    max_angle += np.pi / 5
    sample_error_sum += sample_error
    test_error_sum += test_error
    count += 1

print('Sample Error: %s, Test Error: %s' % (sample_error_sum / count, test_error_sum / count))

# pred = np.concatenate(preds, 0)
# plt.plot(pred[:, 0], pred[:, 1], 'r')

plt.axis('equal')
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
# plt.plot(sample_xy[:, 0], sample_xy[:, 1], 'k*')
# plt.plot(valid_xy[:, 0], pred_y, 'r')
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
plt.plot(degrees, curvature)

plt.subplot(323, projection='polar')
plt.title('curvature (polar)')
plt.plot(theta, curvature)

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
