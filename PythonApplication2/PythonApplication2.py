import numpy as np 
import gym

import time
from functions import SARSA_Learning
                                     
env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env.reset()


alpha = float(input("Enter the value for alpha: "))
gamma = float(input("Enter the value for gamma: "))
epsilon = float(input("Enter the value for epsilon: "))
numberEpisodes = int(input("Enter the number of episodes: "))
 
SARSA1= SARSA_Learning(env,alpha,gamma,epsilon,numberEpisodes)
SARSA1.simulateEpisodes()
SARSA1.computeFinalPolicy()

SARSA1.save_results_to_file('sarsa_results.txt')

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


