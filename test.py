import gym
import os
from ValueTree import *
import numpy as np
import random
import pickle
import os.path

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')

numState = 4

filename = "cartpole_tree.p"
if os.path.isfile(filename):
	f = open(filename, 'rb')
	dataTree = pickle.load(f)
	f.close()
else:
	dataTree = ValueTree(numState)


# print(env.action_space)
print(env.observation_space)

action_pool = [0,1]

total_time = 0

for i_episode in range(1000):
	observation = env.reset()
	for t in range(200):
		env.render()
		pre_observation = observation
		# action = env.action_space.sample()
		if(random.random() < 0.9**i_episode/5):
			action = random.choice(action_pool)
		else:
			action = dataTree.maxAction(observation, action_pool)
		observation, reward, done, info = env.step(action)
		# print(np.append(np.round(pre_observation,1), [action, reward]))
		dataTree.insert(np.round(pre_observation,1), action, reward, observation, action_pool)
		if done:
			print("Episode finished after {} timesteps".format(t+1))
			total_time = total_time + t + 1
			break

print("Average : ", total_time/100.0)
f = open(filename, 'wb')
pickle.dump(dataTree, f)
f.close()