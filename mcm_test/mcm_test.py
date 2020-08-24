import numpy as np
t=[0,      0,   1,   0,    0,   0,   0,   0,   0, 0  ]
y=[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]


def cross_entropy_error(y,t):
    if y.ndim ==1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # print(np.arange(batch_size))
    # print(t)
    # print(y)
    # print(np.arange(batch_size))

    print(y[np.arange(batch_size), t])



    return -np.sum( np.log(         y[np.arange(batch_size),t]      )) / batch_size

print(cross_entropy_error(np.array(y), np.array(t)))