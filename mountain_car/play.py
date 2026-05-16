# A simple python file that runs the Mountain Car game and allow
# user to interact with it using their keyboard!

import gymnasium as gym
from gymnasium.utils.play import play

env = gym.make("MountainCar-v0", render_mode="rgb_array")

mappings = {
    (ord("a"),): 0,
    (ord("d"),): 2
}

play(env, keys_to_action=mappings, fps=30)