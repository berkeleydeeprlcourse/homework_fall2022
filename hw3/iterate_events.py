
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y_best = []
    Y_average = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            print(v)
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Y_best.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y_average.append(v.simple_value)
            
            print('---')
    return X, Y_best, Y_average

logdir1 = "/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_doubledqn_1_LunarLander-v3_12-10-2022_12-42-16/events*"
logdir2 = '/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_doubledqn_2_LunarLander-v3_12-10-2022_14-38-29/events*'
logdir3 = '/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_doubledqn_3_LunarLander-v3_12-10-2022_16-10-18/events*'

# logdir1 = "/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_dqn_1_LunarLander-v3_12-10-2022_11-05-10/events*"
# logdir2 = '/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_dqn_2_LunarLander-v3_12-10-2022_11-07-46/events*'
# logdir3 = '/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_dqn_3_LunarLander-v3_12-10-2022_12-38-59/events*'

X1, Y_best1, Y_average1 = get_section_results(glob.glob(logdir1)[0])
X2, Y_best2, Y_average2 = get_section_results(glob.glob(logdir2)[0])
X3, Y_best3, Y_average3 = get_section_results(glob.glob(logdir3)[0])

#for i, (x, y) in enumerate(zip(X, Y)):
    #print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))


Y_average = np.mean(np.array([Y_average1, Y_average2, Y_average3]), axis=0)

print(Y_average)
X = X1


# logdir1 = "/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_doubledqn_1_LunarLander-v3_12-10-2022_12-42-16/events*"
# logdir2 = '/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_doubledqn_2_LunarLander-v3_12-10-2022_14-38-29/events*'
# logdir3 = '/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_doubledqn_3_LunarLander-v3_12-10-2022_16-10-18/events*'

logdir1 = "/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_dqn_1_LunarLander-v3_12-10-2022_11-05-10/events*"
logdir2 = '/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_dqn_2_LunarLander-v3_12-10-2022_11-07-46/events*'
logdir3 = '/Users/pratikaher/FALL22/homework_fall2022/hw3/data/q2_dqn_3_LunarLander-v3_12-10-2022_12-38-59/events*'

X1, Y_best1, Y_average1 = get_section_results(glob.glob(logdir1)[0])
X2, Y_best2, Y_average2 = get_section_results(glob.glob(logdir2)[0])
X3, Y_best3, Y_average3 = get_section_results(glob.glob(logdir3)[0])

#for i, (x, y) in enumerate(zip(X, Y)):
    #print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))


Y_average2 = np.mean(np.array([Y_average1, Y_average2, Y_average3]), axis=0)

print(Y_average2)
X = X1


print(len(X), len(Y_average))
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
# import seaborn as sns

# plt.plot(X[:49], Y_best[:49], label='Train_BestReturn q1_MsPacman-v0')
plt.plot(X[:49], Y_average[:49], label='Train_AverageReturn q2_MsPacman ddqn')
plt.plot(X[:49], Y_average2[:49], label='Train_AverageReturn q2_MsPacman dqn')

# plt.plot(X[2:102], Y_best[2:102], label='Train_BestReturn q1_MsPacman-v0')
# plt.plot(X[2:102], Y_average[2:102], label='Train_AverageReturn q1_MsPacman')

plt.legend(loc="lower right")

plt.ylabel("Train_Return", fontsize=15)
plt.xlabel("Train_EnvstepsSoFar", fontsize=15, labelpad=4)
plt.title("Learning Curve Question 2 dqn vs ddqn", fontsize=16)
plt.show()



# plt.savefig("/Users/pratikaher/FALL22/homework_fall2022/hw3/figures/q1.png", bbox_inches='tight')
# plt.savefig('/content/gdrive/MyDrive/cs285_f2022/homework_fall2022/hw3/data/q1_MsPacman-v0_12-10-2022_19-17-51/learning_curve.png')