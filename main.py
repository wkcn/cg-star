import numpy as np
import matplotlib.pyplot as plt


# r = 0.5 + 0.2sin(5t)
get_r = lambda theta: 0.5 + 0.2 * np.sin(5 * theta)

# dx/dt = cos(5t)cos(t) - r * sin(t)
get_dx_div_dt = lambda theta: np.cos(5 * theta) * np.cos(theta) - np.sin(theta) * get_r(theta)

# dy/dt = cos(5t)sin(t) + r * cos(t)
get_dy_div_dt = lambda theta: np.cos(5 * theta) * np.sin(theta) + np.cos(theta) * get_r(theta)


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
plt.axis('equal')
plt.axis((-1, 1, -1, 1))

plt.subplot(235)
plt.title('dy/dx')
plt.plot(np.arange(n) * 360 / n, clip_grad)

plt.subplot(236)
plt.title('d2y/dx2')
plt.plot(np.arange(n) * 360 / n, clip_grad2)


plt.show()

plt.subplot(221)
plt.subplot(231, projection='polar')
plt.title('polar')
plt.plot(theta, get_r(theta))

plt.subplot(222)
plt.title('curvature')
K = np.abs(grad2) / np.power(1 + np.square(grad), 1.5)
plt.plot(np.arange(n) * 360 / n, K)

plt.subplot(223, projection='polar')
plt.title('curvature (polar)')
plt.plot(theta, K)

plt.show()
