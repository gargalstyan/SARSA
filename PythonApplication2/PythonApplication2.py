import numpy as np 
import gym

import time
from functions import SARSA_Learning
                                     
env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env.reset()

env.observation_space
  
env.action_space


alpha=0.1
gamma=0.4
epsilon=0.1
numberEpisodes=1000 
 
SARSA1= SARSA_Learning(env,alpha,gamma,epsilon,numberEpisodes)
SARSA1.simulateEpisodes()
SARSA1.computeFinalPolicy()
 
finalLearnedPolicy=SARSA1.learnedPolicy

 
while True:

    env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode='human')
    (currentState,prob)=env.reset()
    env.render()
    time.sleep(2)
    terminalState=False
    for i in range(100):
        if not terminalState:
            (currentState, currentReward, terminalState,_,_) = env.step(int(finalLearnedPolicy[currentState]))
            time.sleep(1)
        else:
            break
    time.sleep(0.5)
env.close()


