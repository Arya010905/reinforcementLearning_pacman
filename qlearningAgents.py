# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
      "You can initialize Q-values here..."
      ReinforcementAgent.__init__(self, **args)
      
      # initializing q vals
      self.qVals = util.Counter() 

    def getQValue(self, state, action):
      """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
      """
      # returns QVal(state, action)
      return self.qVals[state, action]


    def computeValueFromQValues(self, state):
      """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
      """
      # appends all the qvals from actions into vals
      vals = []
      for action in self.getLegalActions(state):
          qVal = self.getQValue(state, action)
          vals.append(qVal)

      # returns max over actions and 0.0 if at terminal state
      if (vals):
          return max(vals)
      else:
          return 0.0

    def computeActionFromQValues(self, state):
      """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
      """
      # initializing actions and val
      actions = self.getLegalActions(state) 
      val = self.getValue(state)
      
      # iterating over actions to check if val is q val to check for best action
      for action in actions:
          if (val == self.getQValue(state, action)):
              return action

    def getAction(self, state):
      """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
      """
      legalActions = self.getLegalActions(state)
      action = None

      # action is random
      if (util.flipCoin(self.epsilon)):
          action = random.choice(legalActions)
      # action is best policy
      else:
          action = self.getPolicy(state)

      return action

    def update(self, state, action, nextState, reward):
      """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
      """
      # computing new q val for update
      val = (1 - self.alpha) * self.getQValue(state, action) 
      val += self.alpha * (reward + (self.discount * self.getValue(nextState)))
      self.qVals[state, action] = val
        
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
      self.featExtractor = util.lookup(extractor, globals())()
      PacmanQAgent.__init__(self, **args)
      self.weights = util.Counter()

    def getWeights(self):
      return self.weights

    def getQValue(self, state, action):
      """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
      """
      # returns q val
      return self.getWeights() * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward: float):
      """
          Should update your weights based on transition
      """
      # initializing qval, features, and diff between expected val and qval
      qVal = 0
      features = self.featExtractor.getFeatures(state, action)
      diff = reward + (self.discount * self.getValue(nextState) - self.getQValue(state, action))
      
      # updating weights
      for feature in features:
        self.weights[feature] += self.alpha * features[feature] * diff
      
    def final(self, state):
      """Called at the end of each game."""
      # call the super-class final method
      PacmanQAgent.final(self, state)

      # did we finish training?
      if self.episodesSoFar == self.numTraining:
          # you might want to print your weights here for debugging
          "*** YOUR CODE HERE ***"
          pass
