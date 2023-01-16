import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler, GridSampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate, \
    plot_intermediate_values
import matplotlib.pyplot as plt
import plotly
import sklearn
from cs285.infrastructure import utils
from cs285.scripts.run_hw2 import PG_Trainer

N_TRIALS = 9  # Maximum number of trials
N_JOBS = -1  # Number of jobs to run in parallel
N_ITER = int(1e2)  # Training budget
TIMEOUT = int(60 * 60 * 3)  # 15 minutes

ENV_ID = "HalfCheetah-v4"
EPISODE_LENGTH = 150
SIZE = 32
DISCOUNT = 0.95

from typing import Any, Dict


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    batch_size = trial.suggest_categorical("batch_size", [10_000, 30_000, 50_000])
    learning_rate = trial.suggest_categorical("learning_rate", [0.005, 0.01, 0.02])

    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }


hyper_params = {
    'env_name': ENV_ID,
    'exp_name': "todo",
    'n_iter': N_ITER,
    'reward_to_go': True,
    'nn_baseline': True,
    'gae_lambda': None,
    'dont_standardize_advantages': False,
    'batch_size': 1000,
    'eval_batch_size': 400,
    'train_batch_size': 1000,
    'num_agent_train_steps_per_iter': 1,
    'discount': DISCOUNT,
    'learning_rate': 5e-3,
    'n_layers': 2,
    'size': SIZE,
    'ep_len': EPISODE_LENGTH,
    'seed': 1,
    'no_gpu': False,
    'which_gpu': 0,
    'video_log_freq': -1,
    'scalar_log_freq': 1,
    'save_params': False,
    'action_noise_std': 0,
    'logdir': None
}


def objective(trial: optuna.Trial) -> float:
    params = hyper_params.copy()
    params['train_batch_size'] = params['batch_size']

    params.update(sample_a2c_params(trial))

    trainer = PG_Trainer(params)
    trainer.rl_trainer.logvideo = False
    trainer.rl_trainer.logmetrics = False

    for itr in range(params["n_iter"]):
        print("\n\n********** Trial %i Iteration %i ************" % (trial._trial_id, itr))

        # collect trajectories, to be used for training
        training_returns = trainer.rl_trainer.collect_training_trajectories(itr, None, trainer.rl_trainer.agent.actor,
                                                                            params['batch_size'])

        paths, envsteps_this_batch, train_video_paths = training_returns

        # add collected data to replay buffer
        trainer.rl_trainer.agent.add_to_replay_buffer(paths)

        # train agent (using sampled data from replay buffer)
        train_logs = trainer.rl_trainer.train_agent()

        if itr > 0 and itr % 10 == 0:
            eval_paths, _ = utils.sample_trajectories(trainer.rl_trainer.env, trainer.rl_trainer.agent.actor,
                                                      params['eval_batch_size'], params['ep_len'])

            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            trial.report(sum(eval_returns), itr)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

    eval_paths, _ = utils.sample_trajectories(trainer.rl_trainer.env, trainer.rl_trainer.agent.actor,
                                              params['eval_batch_size'], params['ep_len'])

    eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

    return sum(eval_returns)


def optimize():
    import torch as th

    # Set pytorch num threads to 1 for faster training
    th.set_num_threads(1)
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = optuna.samplers.GridSampler({
        "batch_size": [10_000, 30_000, 50_000],
        "learning_rate": [0.005, 0.01, 0.02],
    })
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=3)

    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv(f"study_results_pg_{ENV_ID}.csv")

    print(f"best params {study.best_params}")

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig3 = plot_intermediate_values(study)
    fig4 = plot_parallel_coordinate(study)

    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()


if __name__ == '__main__':
    optimize()
