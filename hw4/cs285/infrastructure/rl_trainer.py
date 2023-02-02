import pickle
import time
from collections import OrderedDict

import gym
import numpy as np
import torch
import tqdm

from cs285.agents.mb_agent import MBAgent
from cs285.agents.mbpo_agent import MBPOAgent
# register all of our envs
from cs285.envs import register_envs
from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.utils import sample_trajectories

register_envs()

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
            self.env = gym.make(self.params['env_name'], new_step_api=True)
        else:
            self.env = gym.make(self.params['env_name'], render_mode='rgb_array', new_step_api=True)
        self.env.seed(seed)

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

        if 'sac_params' in self.params['agent_params']:
            self.sac_params = self.params['agent_params']['sac_params']
            self.sac_params['discrete'] = discrete
            self.sac_params['ac_dim'] = ac_dim
            self.sac_params['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif 'render_fps' in self.env.env.metadata:
            self.fps = self.env.env.metadata['render_fps']
        else:
            self.fps = 10

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy, initial_expertdata=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1

        for itr in range(n_iter):
            time_iter = time.time()

            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************" % itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            use_batchsize = self.params['batch_size']
            if itr == 0:
                use_batchsize = self.params['batch_size_initial']

            time_trj = time.time()
            paths, envsteps_this_batch, train_video_paths = (
                self.collect_training_trajectories(
                    itr, initial_expertdata, collect_policy, use_batchsize)
            )

            print(f"Training trajectory collected in {time.time() - time_trj} seconds")

            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            if isinstance(self.agent, MBAgent) or isinstance(self.agent, MBPOAgent):
                self.agent.add_to_replay_buffer(paths, add_sl_noise=self.params['add_sl_noise'])
            else:
                self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")

            time_train = time.time()
            all_logs = self.train_agent()
            print(f"Training completed in {time.time() - time_train} seconds")

            # if doing MBPO, train the model free component
            if isinstance(self.agent, MBPOAgent):
                time_sac_train = time.time()
                for _ in tqdm.tqdm(range(self.sac_params['n_iter']), desc="Sac training loop"):
                    if self.params['mbpo_rollout_length'] > 0:
                        # TODO(Q6): Collect trajectory of length self.params['mbpo_rollout_length'] from the 
                        # learned dynamics model. Add this trajectory to the correct replay buffer.
                        # HINT: Look at collect_model_trajectory and add_to_replay_buffer from MBPOAgent.
                        # HINT: Use the from_model argument to ensure the paths are added to the correct buffer.
                        model_trajectory = self.agent.collect_model_trajectory(self.params['mbpo_rollout_length'])
                        self.agent.add_to_replay_buffer(model_trajectory, True)

                    # train the SAC agent
                    self.train_sac_agent()
                print(f"Sac training completed in {time.time() - time_sac_train} seconds")

            # if there is a model, log model predictions
            if isinstance(self.agent, MBAgent) and itr == 0:
                self.log_model_predictions(itr, all_logs)

            # log/save
            if self.log_video or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))

            print(f"Interation completed in {time.time() - time_iter} seconds")

        print(f"Training loop completed in {time.time() - self.start_time} seconds")

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # TODO: get this from previous HW

        # HINT: depending on if it's the first iteration or not, decide whether to either
        # (1) load the data. In this case you can directly return as follows
        # ``` return loaded_paths, 0, None ```
        if itr == 0 and initial_expertdata:
            with open(initial_expertdata, "rb") as f:
                loaded_paths = pickle.load(f)
                return loaded_paths, 0, None

        # (2) collect `self.params['batch_size']` transitions

        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        t0 = time.time()
        paths, envsteps_this_batch = sample_trajectories(self.env, collect_policy, num_transitions_to_sample, self.params['ep_len'])
        print(f"\nCollected data to be used for training in {time.time() - t0} s")

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        # TODO: get this from previous HW

        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])

            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    def train_sac_agent(self):
        # TODO: Train the SAC component of the MBPO agent.
        # For self.sac_params['num_agent_train_steps_per_iter']:
        # 1) sample a batch of data of size self.sac_params['train_batch_size'] with self.agent.sample_sac
        # 2) train the SAC agent self.agent.train_sac
        # HINT: This will look similar to train_agent above.

        all_logs = []
        for train_step in range(self.sac_params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample_sac(self.params['train_batch_size'])

            train_log = self.agent.train_sac(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################
    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        time_eval_trj = time.time()
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy,
                                                                         self.params['eval_batch_size'],
                                                                         self.params['ep_len'])
        print(f"Eval trajectory collected in {time.time() - time_eval_trj} seconds")

        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
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

            self.logger.flush()

    def log_model_predictions(self, itr, all_logs):
        # model predictions

        import matplotlib.pyplot as plt
        self.fig = plt.figure()

        # sample actions
        action_sequence = self.agent.actor.sample_action_sequences(num_sequences=1, horizon=10)  # 20 reacher
        action_sequence = action_sequence[0]

        # calculate and log model prediction error
        mpe, true_states, pred_states = utils.calculate_mean_prediction_error(self.env, action_sequence,
                                                                              self.agent.dyn_models,
                                                                              self.agent.actor.data_statistics)
        assert self.params['agent_params']['ob_dim'] == true_states.shape[1] == pred_states.shape[1]
        ob_dim = self.params['agent_params']['ob_dim']
        ob_dim = 2 * int(ob_dim / 2.0)  ## skip last state for plotting when state dim is odd

        # plot the predictions
        self.fig.clf()
        for i in range(ob_dim):
            plt.subplot(ob_dim // 2, 2, i + 1)
            plt.plot(true_states[:, i], 'g')
            plt.plot(pred_states[:, i], 'r')
        self.fig.suptitle('MPE: ' + str(mpe))
        self.fig.savefig(self.params['logdir'] + '/itr_' + str(itr) + '_predictions.png', dpi=200, bbox_inches='tight')

        # plot all intermediate losses during this iteration
        all_losses = np.array([log['Training Loss'] for log in all_logs])
        np.save(self.params['logdir'] + '/itr_' + str(itr) + '_losses.npy', all_losses)
        self.fig.clf()
        plt.plot(all_losses)
        self.fig.savefig(self.params['logdir'] + '/itr_' + str(itr) + '_losses.png', dpi=200, bbox_inches='tight')
