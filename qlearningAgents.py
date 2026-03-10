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


import random
import util
import math
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

        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # Counter returns 0.0 for missing keys -> unseen (state, action) pairs have Q=0
        # Return the Q-value for the (state, action) pair
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # Use getQValue exclusively so ApproximateQAgent overrides apply here
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        # Max Q-value among legal actions
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        # Compute the best Q-value
        bestValue = self.computeValueFromQValues(state)
        # Collect all actions achieving the best value for random tie-breaking
        # Unseen actions can be optimal since their Q=0 by default
        bestActions = [action for action in legalActions if self.getQValue(
            state, action) == bestValue]  # Actions with best Q-value
        return random.choice(bestActions)

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
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        if not legalActions:
            return None

        # Epsilon-greedy decision:
        # With probability epsilon, pick a random action (explore)
        # With probability 1-epsilon, pick the best action (exploit)
        if util.flipCoin(self.epsilon):
            # Exploration: choose any legal action at random
            action = random.choice(legalActions)
        else:
            # Exploitation: choose the best action according to Q-values
            action = self.computeActionFromQValues(state)

        # Return the selected action
        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # Q-learning update rule (TD target): r + gamma * max_a' Q(nextState, a')
        sample = reward + self.discount * \
            self.computeValueFromQValues(nextState)
        oldValue = self.getQValue(state, action)
        newValue = (1 - self.alpha) * oldValue + self.alpha * sample
        self.qValues[(state, action)] = newValue

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
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
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
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
        # Question 6 for self practice
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        features = self.featExtractor.getFeatures(state, action)
        q_value = 0.0
        for feature_key, feature_value in features.items():
            q_value += self.weights[feature_key] * feature_value
        return q_value

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        # Question 6 for self practice
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        target = reward + self.discount * \
            self.computeValueFromQValues(nextState)
        prediction = self.getQValue(state, action)
        difference = target - prediction
        # Gradient update for each active feature
        features = self.featExtractor.getFeatures(state, action)
        for feature_key, feature_value in features.items():
            self.weights[feature_key] += self.alpha * \
                difference * feature_value

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

      
        if self.episodesSoFar == self.numTraining:
         
            "*** YOUR CODE HERE ***"
            pass
