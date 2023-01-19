import glob

import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
    min_size = min(len(X), len(Y))
    return X[:min_size], Y[:min_size]


def combine_results(file_name_prefix):
    combined_file_name = f"combined_{file_name_prefix}"

    # logdir = 'data/q2_doubledqn_1_LunarLander-v3*/events*'
    logdir = f'data/{file_name_prefix}*/events*'
    eventfiles = glob.glob(logdir)

    events = [get_section_results(eventfile) for eventfile in eventfiles]
    values = np.empty((len(events), len(events[0][1])))
    labels = np.array(events[0][0])

    for i, event in enumerate(events):
        values[i] = event[1]

    print("File prefix", file_name_prefix)
    for i, (x, y) in enumerate(zip(labels, values)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))

    combined = np.stack([labels, np.mean(values, axis=0), np.std(values, axis=0)])
    np.savetxt(f"{combined_file_name}.txt", combined)

    return combined


if __name__ == '__main__':
    combined_dqn = combine_results("q2_dqn_")
    combined_ddqn = combine_results("q2_doubledqn_")

    file_name = "combined_labels_dqn"
    results = np.loadtxt(f"{file_name}.txt")

    fig = plt.figure()

    plt.axhline(0)
    plt.grid(True)

    plt.errorbar(combined_dqn[0], combined_dqn[1], combined_dqn[2], label="DQN")
    plt.errorbar(combined_ddqn[0], combined_ddqn[1], combined_ddqn[2], label="D-DQN")

    plt.legend()

    plt.savefig("q2.png")
    plt.show()
