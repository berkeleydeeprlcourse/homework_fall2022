import numpy as np


class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        action = self.critic.qa_values(observation)

        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output
        return np.argmax(action.squeeze())