"""
Robocup robot model.

State:
[ x_world ] X position, in world coordinates.
[ y_world ] Y position, in world coordinates.
[   phi   ] Angle of the robot in world coordinates. When phi=0, x is to the
            right of the robot and y is forwards.
[ omega_1 ] Velocity of the first wheel
[ omega_2 ] Velocity of the first wheel
[ omega_3 ] Velocity of the first wheel
[ omega_4 ] Velocity of the first wheel
"""

import numpy as np
import math

# Robot mass, kg
m = 6.35

# Robot radius, m
L = 0.0789

# Gear ratio
n = 60.0 / 20.0

# Robot moment of inertia, kg*m^2
J = 0.5 * m * L ** 2

# Wheel radius, m
r = 0.02711

# Resistance

# Wheel moment of inertia, kg*m^2
Jl = 0.00002516704

# Motor inertia
Jm = 135 * 1e-3 * (1e-2) ** 2

# TODO Wheels are 30 for front and 39 for rear
wheel_angles = [np.deg2rad(60),
                np.deg2rad(129),
                np.deg2rad(-129),
                np.deg2rad(-60)]

# Geometry matrix
# Calculates wheel velocities from generalized velocities.
G = np.asmatrix([[-math.sin(th), math.cos(th), L] for th in wheel_angles]).T / r

# Drag coefficients
c_l = 1e-4
c_m = 1e-3

# Mass matrix
M = np.asmatrix([
    [m, 0, 0],
    [0, m, 0],
    [0, 0, J],
])

Jw = Jl + Jm / n
Q = M + G * Jw * G.T


def rotation_matrix(phi):
    """
    Calculate a rotation matrix for the given angle.
    """
    return np.asmatrix([
        [math.cos(phi), -math.sin(phi), 0],
        [math.sin(phi), math.cos(phi), 0],
        [0, 0, 1]
    ])


def forward_dynamics(x, u, phi):
    """
    Run the continuous model to get dx/dt.

    x: world space velocities
    u: motor torques
    """
    phidot = x[2, 0]
    I4 = np.asmatrix(np.eye(4))
    gRb = rotation_matrix(phi)
    w_m = G.T * n * gRb.T * x

    GTinv = np.linalg.pinv(G.T)
    Ginv = np.linalg.pinv(G)

    Rdot = np.asmatrix([
        [-math.sin(phi), -math.cos(phi), 0],
        [math.cos(phi), -math.sin(phi), 0],
        [0, 0, 0]
    ]) * -phidot

    RTRdot = gRb.T * Rdot

    Z = 0 * (Jm + Jl / n ** 2) * I4 + Ginv * gRb.T * M * gRb * GTinv / n ** 2
    V = 0 * (c_m + c_l / n ** 2) * I4 + Ginv * gRb.T * M * Rdot * GTinv / n ** 2

    wmdot = np.linalg.inv(Z) * (u - V * w_m)

    # X'' = gRb'inv(G.T)rw_l
    return Rdot * GTinv * w_m / n + gRb * GTinv * wmdot / n

def inverse_dynamics(v, a, phi):
    """
    Run the continuous model to get u from dx/dt.

    v: world space velocities
    a: world space accelerations
    """
    phidot = v[2, 0]
    I4 = np.asmatrix(np.eye(4))
    gRb = rotation_matrix(phi)
    w_l = G.T * gRb.T * v

    GTinv = np.linalg.pinv(G.T)
    Ginv = np.linalg.pinv(G)

    Rdot = np.asmatrix([
        [-math.sin(phi), -math.cos(phi), 0],
        [math.cos(phi), -math.sin(phi), 0],
        [0, 0, 0]
    ]) * -phidot

    wdot_l = G.T * gRb.T * (a - Rdot * GTinv * w_l)

    RTRdot = gRb.T * Rdot

    Z = (Jm + Jl / n ** 2) * I4 + Ginv * gRb.T * M * gRb * GTinv / n ** 2
    V = (c_m + c_l / n ** 2) * I4 + Ginv * gRb.T * M * Rdot * GTinv / n ** 2

    w_m = w_l * n
    wmdot = wdot_l * n
    return Z * wmdot + V * w_m

def main():
    """Run the program."""
    v = np.asmatrix([[0.2, 0.2, 1]]).T
    a = np.asmatrix([[1, 2, 3]]).T
    xdot = forward_dynamics(v, inverse_dynamics(v, a, 0), 0)
    print(xdot)
    Ginv = np.linalg.pinv(G)


if __name__ == '__main__':
    main()
