import glob

import numpy as np
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    E = []
    T = []
    time0 = None
    for e in summary_iterator(file):
        if time0 is None:
            time0 = e.wall_time
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
                T.append(int(e.wall_time - time0))
                time0 = e.wall_time
            elif v.tag == 'Eval_StdReturn':
                E.append(v.simple_value)
    return X, Y, E, T


if __name__ == '__main__':
    import glob

    logdir = 'data/hw4_q3_reacher_reacher-cs285-v0_26-01-2023_21-50-00'
    eventfile = glob.glob(logdir + "/events*")[0]

    X, Y, E, T = get_section_results(eventfile)
    for i, (x, y, e, t) in enumerate(zip(X, Y, E, T)):
        print('{} Iteration {:d} | Train steps: {:d} | Return: {}, Std {}'.format(t, i, int(x), y, e))

    np.savetxt(logdir + "/eval.txt", np.array([X, Y, E, T]))
