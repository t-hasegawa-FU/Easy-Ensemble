import numpy as np

# Target shape is x [bs, ch, winsize].

class jitter:
    def __init__(self, m=1.0):
        self.m = m
    def __call__(self, x, y):
        sigma = self.m / 50
        jit = np.float32(np.random.normal(0, sigma, x.shape))
        return (x + jit, y)

class scale:
    def __init__(self, m=1.0):
        self.m = m
    def __call__(self, x, y):
        sigma = self.m / 2.5
        scale = np.random.normal(1, sigma)
        return (x * scale, y)

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)

class rotation:
    def __init__(self, m=3):
        self.m = m
    def __call__(self, x, y):
        # swap the channels by each m
        idx = np.arange(x.shape[1]).reshape(-1, self.m)
        idx = shuffle_along_axis(idx, axis=1).reshape(-1)
        return x[:, idx, :], y

class flip:
    def __call__(self, x, y):
        # flip the positive/negative for each axis (m is not used)
        b = np.array([-1,1], dtype=np.float32)
        x2 = x.copy() * b[np.random.randint(2, size=x.shape[1])].reshape(-1,x.shape[1], 1)
        return x2, y

class rotflp:
    def __init__(self, m=3):
        self.rot = rotation(m=m)
    def __call__(self, x, y):
        x2, _ = self.rot(x, y)
        x2, _ = flip()(x2, y)
        return x2, y

class inverse:
    def __call__(self, x, y):
        return x[..., ::-1], y

class vershift:
    def __init__(self, m=1.0):
        self.m = m
    def __call__(self, x, y):
        sigma = self.m / 5
        w = np.random.normal(0, sigma, size=x.shape[1]).reshape(1, x.shape[1], 1)
        return x + w, y

class holshift:
    def __init__(self, m=1.0):
        self.m = m
    def __call__(self, x, y):
        num = int(x.shape[2] * self.m / 11)
        if np.random.random() < 0.5:
            return np.concatenate([np.repeat(x[..., :1], num, axis=2), x[..., :-num]], axis=2), y
        else:
            return np.concatenate([x[..., num:], np.repeat(x[..., -1:], num, axis=2)], axis=2), y
