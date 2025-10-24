import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from copy import deepcopy
from functools import partial
from collections import deque
import argparse
import jax

import sys
sys.path.append('.')
from dsynth.web_utils import WebsocketPolicyServer

from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper, ResizeImageWrapper, TemporalEnsembleWrapper, stack_and_pad
from octo.utils.train_callbacks import supply_rng

def preprocess_raw_obs(raw_obs):
    new_obs = {}
    new_obs['image_primary'] = raw_obs['observation/left_base_camera_link']
    new_obs['image_secondary'] = raw_obs['observation/right_base_camera_link']
    new_obs['image_wrist'] = raw_obs['observation/fetch_hand']
    return new_obs

class HistoryBuffer:
    def __init__(self, horizon: int):
        self.horizon = horizon

        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

    def get_obs(self, new_obs):
        self.num_obs += 1
        self.history.append(new_obs)
        if len(self.history) < self.horizon:
            # pad history buffer with the first item
            sample_obs = self.history.pop()
            self.history.clear()
            self.history.extend([sample_obs] * self.horizon)
        assert len(self.history) == self.horizon
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs
    
class RHC:
    def __init__(self, exec_horizon: int):
        self.exec_horizon = exec_horizon
    
    def process(self, actions):
        return actions[:self.exec_horizon]


class Policy:
    def __init__(self, octo_model, horizon=4, history=2, exec_horizon=1):
        self.model = octo_model
        self.horizon = horizon
        self.history = history
        self.rhc = RHC(exec_horizon)
        self.history_buffer = None
        self.task = None

        self.policy_fn = supply_rng(
            partial(
                self.model.sample_actions,
                unnormalization_statistics=self.model.dataset_statistics["action"],
            ),
        )

    def infer(self, obs_raw):
        if obs_raw['time_step'] == 0:
            self.history_buffer = HistoryBuffer(self.history)
            self.task = self.model.create_tasks(texts=[obs_raw['prompt']])
        obs = preprocess_raw_obs(obs_raw)
        obs = self.history_buffer.get_obs(obs)

        actions = self.policy_fn(jax.tree_map(lambda x: x[None], obs), self.task)
        actions = np.array(actions[0])

        actions = self.rhc.process(actions)
        if actions.ndim == 1:
            actions = actions[None, :]
        
        result = {
            "actions": actions
        }
        return result


def main(args):
    octo_model = OctoModel.load_pretrained(args.finetuned_path)
    policy = Policy(octo_model)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
    )
    server.serve_forever()



def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument("-e", "--env-id", type=str, default="PickToCartEnv", help=f"Environment to run motion planning solver on. Available options are {list(MP_SOLUTIONS.keys())}")

    parser.add_argument("--finetuned-path", type=str)
   
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=8000)
    
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())