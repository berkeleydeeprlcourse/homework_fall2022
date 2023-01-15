from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
from matplotlib import pyplot as plt

from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.action_noise_wrapper import ActionNoiseWrapper
from cs285.infrastructure.utils import sample_trajectories

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        # seed = None

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        if self.params['video_log_freq'] == -1:
            render_mode = None
        else:
            render_mode = 'rgb_array'
        self.env = gym.make(self.params['env_name'], render_mode=render_mode, new_step_api=True)
        #if seed is not None:
            #self.env.seed(seed)

        # Add noise wrapper
        if params['action_noise_std'] > 0:
            self.env = ActionNoiseWrapper(self.env, seed, params['action_noise_std'])

        # import plotting (locally if 'obstacles' env)
        if not (self.params['env_name'] == 'obstacles-cs285-v0'):
            import matplotlib
            matplotlib.use('Agg')

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30  # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10

        self.val_means = []
        self.val_stds = []
        self.loss = []
        self.baseline_loss = []

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for _ in range(1):
            self.val_means.append([])
            self.val_stds.append([])
            self.loss.append([])
            self.baseline_loss.append([])

            for itr in range(n_iter):
                print("\n\n********** Iteration %i ************" % itr)

                # decide if videos should be rendered/logged at this iteration
                if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                    self.logvideo = True
                else:
                    self.logvideo = False

                # decide if metrics should be logged
                if self.params['scalar_log_freq'] == -1:
                    self.logmetrics = False
                elif itr % self.params['scalar_log_freq'] == 0:
                    self.logmetrics = True
                else:
                    self.logmetrics = False

                # collect trajectories, to be used for training
                training_returns = self.collect_training_trajectories(itr,
                                                                      initial_expertdata, collect_policy,
                                                                      self.params['batch_size'])
                paths, envsteps_this_batch, train_video_paths = training_returns
                self.total_envsteps += envsteps_this_batch

                # add collected data to replay buffer
                self.agent.add_to_replay_buffer(paths)

                # train agent (using sampled data from replay buffer)
                train_logs = self.train_agent()

                # log/save
                if self.logvideo or self.logmetrics:
                    # perform logging
                    print('\nBeginning logging procedure...')
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, train_logs)

                    if self.params['save_params']:
                        self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

        self.plot_chart()

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, load_initial_expertdata, collect_policy, batch_size):
        # TODO: GETTHIS from HW1

        """
                :param itr:
                :param load_initial_expertdata:  path to expert data pkl file
                :param collect_policy:  the current policy using which we collect data
                :param batch_size:  the number of transitions we collect
                :return:
                    paths: a list trajectories
                    envsteps_this_batch: the sum over the numbers of environment steps in paths
                    train_video_paths: paths which also contain videos for visualization purposes
                """

        # TODO decide whether to load training data or use the current policy to collect more data
        # HINT: depending on if it's the first iteration or not, decide whether to either
        # (1) load the data. In this case you can directly return as follows
        # ``` return loaded_paths, 0, None ```
        if itr == 0 and load_initial_expertdata:
            with open(load_initial_expertdata, "rb") as f:
                loaded_paths = pickle.load(f)
                return loaded_paths, 0, None

        # (2) collect `self.params['batch_size']` transitions

        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.logvideo:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        # TODO: GETTHIS from HW1

        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['train_batch_size'])

            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy,
                                                                         self.params['eval_batch_size'],
                                                                         self.params['ep_len'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            # save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.val_means[-1].append(logs["Eval_AverageReturn"])
            self.val_stds[-1].append(logs["Eval_StdReturn"])
            self.loss[-1].append(logs["Training Loss"])

            if "Baseline Training Loss" in logs:
                self.baseline_loss[-1].append(logs["Baseline Training Loss"])

            self.logger.flush()

    def plot_chart(self):
        interval = 5
        exp_name = self.params["exp_name"]

        plot_baseline_loss = self.baseline_loss is not None and len(self.baseline_loss) > 0 and len(
            self.baseline_loss[0]) > 0

        fig, ax = plt.subplots(1, 3 if plot_baseline_loss else 2, figsize=(20, 10))
        fig.suptitle("q1_sb_no_rtg_dsa - Cartpole-V0")
        fig.supxlabel("Network with 2 Hidden Layers of size 64")

        y = np.array(self.loss)
        y = np.mean(y, axis=0)
        x = np.array(range(len(y)))

        ax1 = ax[0]
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title("Iteration vs Training Loss")
        ax1.set_xticks(np.arange(0, len(x) + 1, interval))
        ax1.plot(x, y)

        y = np.array(self.val_means)
        y = np.mean(y, axis=0)
        y_std = np.array(self.val_stds)
        y_std = np.mean(y_std, axis=0)
        x = np.array(range(len(y)))

        np.savetxt(f"{exp_name}.txt", y)

        ax3 = ax[1]
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Mean Rewards')
        ax3.set_title("Iteration vs Mean Rewards")
        print(np.arange(0, len(x) + 1, 50))
        ax3.set_xticks(np.arange(0, len(x) + 1, interval))
        ax3.errorbar(x, y, yerr=y_std, label="Mean Rewards", ecolor='r')
        # ax2.axhline(y=5567, color='r', linestyle='-', label="Expert")
        # ax2.axhline(y=138, color='g', linestyle='-', label="Behaviour Cloning")
        ax3.legend()

        if plot_baseline_loss:
            y = np.array(self.baseline_loss)
            y = np.mean(y, axis=0)
            x = np.array(range(len(y)))

            ax2 = ax[2]
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss')
            ax2.set_title("Iteration vs Training Baseline Loss")
            ax2.set_xticks(np.arange(0, len(x) + 1, interval))
            ax2.plot(x, y)

        plt.savefig(f"{exp_name}.png")
