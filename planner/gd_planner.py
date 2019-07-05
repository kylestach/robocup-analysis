import numpy as np
import scipy.optimize as opt

obstacles = [np.asmatrix([[0, 3]]).T, np.asmatrix([[3, 8]]).T]

"""
Variables looks like this:
[ p1x ]
[ p1y ]
[ v1x ]
[ v1y ]
[ p2x ]
[ p2y ]
[ v2x ]
[ v2y ]
[ ... ]
There are four entries per waypoint and n+1 waypoints
(including the start and end). However, the first and last waypoints only have
one value (norm of the tangent vector).
So the vector length is 4n-2
"""

def hermite_spline(p0x, p0y, t0x, t0y, p1x, p1y, t1x, t1y, s):
    return np.asarray([
        (2 * s ** 3 - 3 * s ** 2 + 1) * p0x +
        (s ** 3 - 2 * s ** 2 + s) * t0x +
        (-2 * s ** 3 + 3 * s ** 2) * p1x +
        (s ** 3 - s ** 2) * t1x,
        (2 * s ** 3 - 3 * s ** 2 + 1) * p0y +
        (s ** 3 - 2 * s ** 2 + s) * t0y +
        (-2 * s ** 3 + 3 * s ** 2) * p1y +
        (s ** 3 - s ** 2) * t1y
    ])

def hermite_deriv(p0x, p0y, t0x, t0y, p1x, p1y, t1x, t1y, s):
    return np.asarray([
        (6 * s ** 2 - 6 * s) * p0x +
        (3 * s ** 2 - 4 * s + 1) * t0x +
        (-6 * s ** 2 + 6 * s) * p1x +
        (3 * s ** 2 - 2 * s) * t1x,
        (6 * s ** 2 - 6 * s) * p0y +
        (3 * s ** 2 - 4 * s + 1) * t0y +
        (-6 * s ** 2 + 6 * s) * p1y +
        (3 * s ** 2 - 2 * s) * t1y
    ])

def hermite_deriv2(p0x, p0y, t0x, t0y, p1x, p1y, t1x, t1y, s):
    return np.asarray([
        (12 * s - 6) * p0x +
        (6 * s - 4) * t0x +
        (-12 * s + 6) * p1x +
        (6 * s - 2) * t1x,
        (12 * s - 6) * p0y +
        (6 * s - 4) * t0y +
        (-12 * s + 6) * p1y +
        (6 * s - 2) * t1y
    ])


def hermite_curvature(p0x, p0y, t0x, t0y, p1x, p1y, t1x, t1y, s):
    d = hermite_deriv(p0x, p0y, t0x, t0y, p1x, p1y, t1x, t1y, s)
    d2 = hermite_deriv2(p0x, p0y, t0x, t0y, p1x, p1y, t1x, t1y, s)
    return (d[0] * d2[1] - d[1] * d2[0]) / (np.linalg.norm(d) ** 3)


def evaluate_spline(coefficients, boundary, s):
    """
    args
        coefficients: array shaped as above
        boundary: array [p0x, p0y, t0x, t0y, p1x, p1y, t1x, t1y]
    """
    num_segments = (coefficients.shape[0] + 4) / 4
    spline_index = int(s * num_segments)
    if spline_index == num_segments and spline_index > 0:
        spline_index -= 1
    s_local = s * num_segments - spline_index

    p0x, p0y = 0., 0.
    t0x, t0y = 0., 0.
    p1x, p1y = 0., 0.
    t1x, t1y = 0., 0.

    idx_0, idx_1 = 4 * (spline_index - 1), 4 * spline_index
    if spline_index == 0:
        p0x, p0y = boundary[0], boundary[1]
        t0x, t0y = boundary[2], boundary[3]
    else:
        p0x, p0y = coefficients[idx_0], coefficients[idx_0 + 1]
        t0x, t0y = coefficients[idx_0 + 2], coefficients[idx_0 + 3]

    if spline_index >= num_segments - 1:
        p1x, p1y = boundary[4], boundary[5]
        t1x, t1y = boundary[6], boundary[7]
    else:
        p1x, p1y = coefficients[idx_1], coefficients[idx_1 + 1]
        t1x, t1y = coefficients[idx_1 + 2], coefficients[idx_1 + 3]

    return (hermite_spline(p0x, p0y, t0x, t0y, p1x, p1y, t1x, t1y, s_local),
            hermite_deriv2(p0x, p0y, t0x, t0y, p1x, p1y, t1x, t1y, s_local))


boundary = np.array([0, 0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
obstacles = [np.array([0.25, 0.75]), np.array([1.0, 0.3])]


def obstacle_cost(p, o):
    return 24 * np.exp(
        -(np.linalg.norm(p - o) ** 2 / (2 * 0.2))) / len(obstacles)


def cost(a):
    total = 0.0
    last_position = boundary[:2]
    for s in np.linspace(0, 1, 5):
        position, acceleration = evaluate_spline(a, boundary, s)
        total += np.linalg.norm(acceleration) ** 2
        last_position = position
        for obstacle in obstacles:
            total += obstacle_cost(position, obstacle)
    return total

def main():
    coeffs = np.array([
        0.6,
        -0.3,
        2.0,
        0.5,
    ])
    res = opt.minimize(cost, coeffs)
    for s in np.linspace(0, 1, 0.5):
        position, _ = evaluate_spline(res.x, boundary, s)
        print("%s, %s" % (position[0], position[1]))
    print(res.fun)
    print(res.x)

if __name__ == '__main__':
    main()
