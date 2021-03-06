import gym
import os
from ValueTree import *
import base
import numpy as np
import random
import pickle
import os.path

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
limit_high = env.observation_space.high
limit_low = env.observation_space.low

numState = 4

filename = "cartpole_net_test.p"
if os.path.isfile(filename):
	f = open(filename, 'rb')
	cartpole_net = pickle.load(f)
	f.close()
else:
	cartpole_net = base.NN(4,2,[32], func='lrelu', weight='xavier', dropout=0.8)

# Parameters for neural network 
total_time = 0

pass_score = 75
num_pass = 0
epoch = 5

TRAINING = False

num_episodes = 100000
if not TRAINING:
	num_episodes = 10

for i_episode in range(num_episodes):
	input_array = []
	output_array = []
	# reset variables for a new episode 
	observation = env.reset()
	observation = [x for x in observation]
	# observation = [(x-limit_low[i])/(limit_high[i]-limit_low[i]) for i, x in enumerate(observation)]
	sum_reward = 0
	for t in range(200):
		if not TRAINING:
			env.render()
		pre_observation = observation
		if TRAINING:
			action = env.action_space.sample()
		else:
			# print(cartpole_net.forward([x for x in pre_observation]))
			net_output = cartpole_net.forward(pre_observation)
			print(pre_observation, net_output)
			action = 0 if net_output[0][0]>net_output[0][1] else 1
		input_array.append(pre_observation)
		if(action==0):
			output_array.append([1,0])
		else:
			output_array.append([0,1])
		observation, reward, done, info = env.step(action)
		observation = [x for x in observation]
		# observation = [(x-limit_low[i])/(limit_high[i]-limit_low[i]) for i, x in enumerate(observation)]
		sum_reward += reward
		if done: 
			if TRAINING and (t+1 >= pass_score):
			# if t+1 >= pass_score:
				num_pass = num_pass + 1
				for i in range(len(input_array)):
					for _ in range(epoch):
						cartpole_net.train(input_array[i], output_array[i],0.001)
				# if(len(input_array)!=0):
				# 	rand_order = np.random.permutation(len(input_array))
				# 	for idx in rand_order:
				# 		for _ in range(epoch):
				# 			cartpole_net.train(input_array[idx], output_array[idx],0.001)
				print("Episode",i_episode,"finished after {} timesteps".format(t+1))
			total_time = total_time + t + 1
			break

if TRAINING:
	print("Number of training cases: ", num_pass)
	f = open(filename, 'wb')
	pickle.dump(cartpole_net, f)
	f.close()
else:
	print("Average : ", total_time/num_episodes)
	