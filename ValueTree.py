#!/usr/bin/python

import numpy as np
import math
import random

class ValueTree(object):

	def __init__(self, numState):
		self.root = ValueNode('root')
		self.numState = numState
		self.alpha = 0.1 # learning rate
		self.gamma = 0.9 # discounting factor
		self.root.alpha = self.alpha
		self.root.gamma = self.gamma

	''' 
	insert the given list into the tree
	List includes action and state value (len = numState + 2)
	'''
	def insert(self, cur_state, action, reward, next_state, action_pool):
		data = np.append(cur_state, [action, reward])
		if(len(data)!=self.numState+2):
			print("Inappropriate data!")
		else:
			self.root.makeBranch(self.root, data, action, reward, next_state, action_pool)

	'''
	Find the branch with given data and return the value at the end
	data includes action (len = numState + 1)
	'''
	def find(self, data):
		if(len(data)!=self.numState+1):
			print("Not enough values given to find!")
			return None
		else:
			return self.root.searchBranch(data)

	def maxAction(self, state, action_pool):
		if(len(state)!=self.numState):
			print("Nor snough states given!")
			return None
		return self.root.getMaxAction(state, action_pool)



class ValueNode(object):

	'''
	initialize new node object
	value : value of the node
	child : list containing the children of the node. 
	'''
	def __init__(self, value):
		self.children = []
		self.value = value
		self.alpha = 0
		self.gamma = 0

	'''
	add(child_node):
	adds given child node into the tree structure
	child_node : ValueNode to be added into children list
	'''
	def addNode(self, child_node):
		self.children.append(child_node)

	def makeBranch(self, root, data, action, reward, next_state, action_pool):
		if(len(data)>1):
			childNode = self.searchChild(data[0])
			if(childNode==None):
				childNode = ValueNode(data[0])
				self.addNode(childNode)
			childNode.makeBranch(root, data[1:], action, reward, next_state, action_pool)
		if(len(data)==1):
			if(len(self.children)>1):
				print("Error! incorrect branch made")
			elif(len(self.children)==1):
				self.children[0].updateVal(root, data[0], next_state, action_pool)
			else:
				self.addNode(ValueNode(data[0]))


	def searchBranch(self, data):
		childNode = self.searchChild(data[0])
		if(childNode==None):
			return None

		if(len(data)==1):
			return childNode.children[0].value
		if(len(data)>1):
			return childNode.searchBranch(data[1:])

	def getMaxAction(self, state, action_pool):
		childNode = self.searchChild(state[0])
		if(childNode==None):
			return random.choice(action_pool)

		if(len(state)==1):
			maxAction = random.choice(action_pool)
			maxValue = childNode.searchChild(maxAction)
			if(maxValue==None): maxValue = 0
			for a in action_pool:
				actionNode = childNode.searchChild(a)
				if(actionNode!=None and actionNode.value > maxValue): 
					maxValue = actionNode.value
					maxAction = a
			return maxAction
		if(len(state)>1):
			return childNode.getMaxAction(state[1:], action_pool)

	''' 
	changeVal(newVal)
	Changes the value of the node with the new given newVal
	newVal : new value of the node. Type int, float, string, etc
	'''
	def changeVal(self, newVal):
		self.value = newVal

	'''
	update the current value using the new value according to the defined value update rule
	'''
	def updateVal(self, root, reward, next_state, action_pool):
		#  TODO: add the value update method
		maxQ = -math.inf 
		for a in action_pool:
			nextQ = root.searchBranch(np.append(next_state, [a]))
			if (nextQ==None): nextQ = 0
			if (nextQ > maxQ): maxQ = nextQ

		self.value = self.value + root.alpha * (reward + root.gamma * maxQ - self.value)

		# Q(s,a) = Q(s,a) + alpha * (reward + gamma * maxQ(s_next,a_max) - Q(s,a))

	'''
	Searchs for the child with given value. return None if not found
	'''
	def searchChild(self, value):
		# self.quickSearch(self.child)
		for i in range(len(self.children)):
			if(self.children[i].value == value):
				return self.children[i]
		return None

	'''
	quick search of given list
	'''
	def quickSearch(self, list):
		return None # TODO: dummy code to be replaced
