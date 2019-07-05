import numpy as np
import matplotlib.pyplot as plt

class Path:
    def __init__(self):
        pass

    def eval(self, t):
        return np.dot(np.array([
            [0, 0, 3, -2],
            [0, 1, -2, 2],
            [0, 0, 0, 0]
        ]), np.array([1, t, t ** 2, t ** 3]))

def main():
    ts = np.linspace(0, 1, 101)
    spline = Path()
    vals = spline.eval(ts)
    plt.plot(vals[1], vals[0])
    plt.show()

if __name__ == '__main__':
    main()
