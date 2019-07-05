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

"""
Optimize over the following space:
( Constant: p0, angle(t0, tn) )
( Implicit: (p, t)(1..n) )

[ |t0| ]
[ a0x ]
[ j0x ]
[ a0y ]
[ j0y ]
[ a1x ]
[ j1x ]
[ a1y ]
[ j1y ]
[ a2x ]
[ j2x ]
[ a2y ]
[ j2y ]
[ ... ]
[ anx ]
[ jnx ]
[ any ]
[ jny ]
[|t{n+1}|]
"""

def eval_spline_cost(parameters):
    """
    parameters: 2 x 4 matrix
    """
    for t in np.linspace(0, 0.99, 100):
        # Calculate curvature

        # Calculate ds
        ds = hermite_deriv(

def eval_path_cost(parameters):
    for p
