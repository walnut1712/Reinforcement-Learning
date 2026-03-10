# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2a():
    """
      Prefer the close exit (+1), risking the cliff (-10).
    """
    # answerDiscount = None
    # answerNoise = None
    # answerLivingReward = None

    # Rationale:
    # - Low discount (0.2) emphasizes short-term rewards, preferring the nearby +1 exit.
    # - Zero noise (0.0) removes slippage so risky path is acceptable.
    # - Zero living reward (0.0) is neutral about time steps.
    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2b():
    """
      Prefer the close exit (+1), but avoiding the cliff (-10).
    """
    # answerDiscount = None
    # answerNoise = None
    # answerLivingReward = None

    # Rationale:
    # - Keep discount low (0.2) to still prefer the close exit.
    # - Introduce some noise (0.2) so the optimal path avoids states near the cliff.
    # - Zero living reward to avoid incentivizing wandering.
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2c():
    """
      Prefer the distant exit (+10), risking the cliff (-10).
    """
    # answerDiscount = None
    # answerNoise = None
    # answerLivingReward = None

    # Rationale:
    # - Higher discount (0.9) values the larger +10 reward despite longer path.
    # - Zero noise makes the risky shorter route viable.
    # - Zero living reward is neutral about step cost.
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2d():
    """
      Prefer the distant exit (+10), avoiding the cliff (-10).
    """
    # answerDiscount = None
    # answerNoise = None
    # answerLivingReward = None

    # Rationale:
    # - High discount (0.9) still prefers +10 overall.
    # - Non-zero noise (0.2) discourages paths that brush the cliff, pushing a safer route.
    # - Zero living reward avoids bias toward stalling.
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question2e():
    """
      Avoid both exits and the cliff (so an episode should never terminate).
    """
    # answerDiscount = None
    # answerNoise = None
    # answerLivingReward = None

    # Rationale:
    # - Positive living reward (1.0) makes continuing forever better than any terminal payoff.
    # - Discount is high (0.9) to not devalue future living rewards too fast.
    # - Zero noise is fine; the key is positive living reward.
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 1.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
