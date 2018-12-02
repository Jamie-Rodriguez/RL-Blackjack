import gym
import random
import numpy as np
from collections import defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import pandas as pd


def epsilon_greedy_policy(epsilon, actionValues):
    """
    Get an action to take using the Epsilon-Greedy policy
    Take a random action "epsilon" percent of the time
    Otherwise take the best action (the action with the highest value)

    Parameters
    ----------
    epsilon : float
        0 <= epsilon <= 1
        Percentage of the time to take a random action in "actionValues"

    actionValues : numpy.ndarray
        1D array of type float
        The possible actions to take given a state, where the array is indexed by action

    Returns
    -------
    int
        The index of the action to take
    """
    p = np.random.random()

    return np.argmax(actionValues) if p < (1 - epsilon) else random.randint(0, len(actionValues) - 1)


def getStatesActionsRewardsFromPlayingGame(env, epsilon, Q):
    """
    Play a game and get the actions, states and rewards for each step of the game

    Parameters
    ----------
    env : gym.envs
        The OpenAI Gym environment used to play the game

    epsilon : float
        0 <= epsilon <= 1
        Epsilon value used in the Epsilon-Greedy policy when choosing an action
        Percentage of the time to take a random action

    Q : defaultdict(tuple: numpy.ndarray)
        Q-value dictionary to base the policy of the game on (uses the Epsilon-Greedy policy)

        Structure of dictionary:
            key : tuple
                state of game (observation from "env")
            value : numpy.ndarray
                1D array of type float
                array of possible actions in state, indexed by action

        "Q" is a defaultdict that creates an array of actions if state hasn't yet been visited

    Returns
    -------
    array_like(tuple, int, float)
        Array-like object of triples: (states, actions, rewards) from game
        Be aware of the timing:
            each triple is s(t), a(t), r(t)
            but r(t) results from taking action a(t-1) from s(t-1) and landing in s(t)
    """
    done = False

    s = env.reset()
    a = epsilon_greedy_policy(epsilon, Q[s])
    statesActionsRewards = [(s, a, 0)]

    while not done:
        s, r, done, info = env.step(a)

        if done:
            statesActionsRewards.append((s, None, r))
            break
        else:
            a = epsilon_greedy_policy(epsilon, Q[s])
            statesActionsRewards.append((s, a, r))

    return statesActionsRewards


def calculateReturns(gamma, statesActionsRewards):
    """
    Calculate the returns from the states, actions and rewards of a game
    where returns are given by:
        G[t] = r[t+1] + gamma * G[t+1]
    The value of the terminal state is 0 by definition
    Ignore first G because it is meaningless as it doesn't correspond to any move

    Parameters
    ----------
    gamma : float
        0 <= gamma <= 1
        The discount factor of future rewards

    statesActionsRewards : array_like(tuple, int, float)
        Array-like object of triples: (states, actions, rewards) to calculate returns from

    Returns
    -------
    array_like(tuple, int, float)
        Array-like object of states, actions, returns from game
    """
    G = 0
    statesActionsReturns = []
    first = True
    for s, a, r in reversed(statesActionsRewards):
        if first:
            first = False
        else:
            statesActionsReturns.append((s, a, G))
        G = r + gamma * G

    statesActionsReturns.reverse()

    return statesActionsReturns


def playEpisode(env, epsilon, gamma, Q):
    """
    Plays a game and calculates the returns from the episode

    Parameters
    ----------
    env : gym.envs
        The OpenAI Gym environment used to play the game

    epsilon : float
        0 <= epsilon <= 1
        Epsilon value used in the Epsilon-Greedy policy when choosing an action
        Percentage of the time to take a random action

    gamma : float
        0 <= gamma <= 1
        The discount factor of future rewards

    Q : defaultdict(tuple: numpy.ndarray)
        Q-value dictionary to base the policy of the game on (uses the Epsilon-Greedy policy)

        Structure of dictionary:
            key : tuple
                state of game (observation from "env")
            value : numpy.ndarray
                1D array of type float
                array of possible actions in state, indexed by action

        "Q" is a defaultdict that creates an array of actions if state hasn't yet been visited

    Returns
    -------
    array_like(tuple, int, float)
        Array-like object of states, actions, returns from game
    """
    statesActionsRewards = getStatesActionsRewardsFromPlayingGame(env, epsilon, Q)

    return calculateReturns(gamma, statesActionsRewards)


def updateQValues(Q, alpha, statesActionsReturns):
    """
    Updates the Q-values of "Q" using the states, actions and returns from an episode
    Update is according to the following formula:
        Q[s][a] <- Q[s][a] + alpha * (G - Q[s][a])

    Note
    ----
    This function mutates Q!
    Can't use a functional approach because making copies of Q here is very slow when calling this function multiple times

    Parameters
    ----------
    Q : defaultdict(tuple: numpy.ndarray)
        Q-value dictionary to update using the states, actions and returns from an episode

        Structure of dictionary:
            key : tuple
                state of game
            value : numpy.ndarray
                1D array of type float
                array of possible actions in state, indexed by action

        "Q" is a defaultdict that creates an array of actions if state hasn't yet been visited

    alpha : float
        0 <= alpha <= 1
        Learning rate when updating Q

    statesActionsReturns : array_like(tuple, int, float)
        Array-like object of states, actions, returns from an episode used to update Q-values in Q

    Returns
    -------
    defaultdict(tuple: numpy.ndarray)
        Updated version of "Q" using the states, actions and returns of an episode
    """
    seenStateActionPairs = set()

    for s, a, G in statesActionsReturns:
        sa = (s, a)
        if sa not in seenStateActionPairs:
            Q[s][a] = Q[s][a] + alpha * (G - Q[s][a])

            seenStateActionPairs.add(sa)

    return Q


def plotRunningAverage(totalReturns, windowSize):
    """
    Plots the running average of the total returns of the episodes against the number of episodes run

    Parameters
    ----------
    totalReturns : array_like
        1D array of floats
        Array-like structure of the total returns of each episode

    windowSize : int
        The size of the window of the running average
    """
    # Reduce the number of points used in the plot
    numberOfPointsToDisplay = 1000

    N = len(totalReturns)
    runningAverage = np.empty(N)

    for n in range(N):
        runningAverage[n] = totalReturns[max(0, n - int(windowSize)):(n + 1)].mean()

    # make sure number of points to display <= number of games played
    if numberOfPointsToDisplay > N:
        numberOfPointsToDisplay = N

    stepSize = int(N/numberOfPointsToDisplay)
    numberOfIterations = np.arange(0, N, stepSize)
    runningAverageDisplay = runningAverage[0::stepSize]

    mean = runningAverage.mean()
    std = runningAverage.std()
    figureSize = (8, 6)
    xLabel = "Number of Games Played"
    yLabel = "Running Average of Score"
    title = f"Number of Games vs Average Score\n(mean: {mean:.2f}, std: {std:.2f})"

    plt.figure("Linear Plot", figsize=figureSize)
    plt.plot(numberOfIterations, runningAverageDisplay)
    plt.xlabel(xLabel)
    plt.xlim(1, N)
    plt.ylabel(yLabel)
    plt.ylim(-1, 1)
    plt.title(title)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(f"Results-Linear-{N}.png", bbox_inches="tight")

    plt.figure("Logarithmic Plot", figsize=figureSize)
    plt.semilogx(numberOfIterations, runningAverageDisplay, nonposx="clip")
    plt.xlabel(xLabel)
    plt.xlim(1, N)
    plt.ylabel(yLabel)
    plt.ylim(-1, 1)
    plt.title(title)
    plt.savefig(f"Results-Logarithmic-{N}.png", bbox_inches="tight")


def monteCarlo(env, epsilon, gamma, alpha, numEpisodes, initialQ):
    """
    Starting with the Q-values defined by "initialQ", plays through "numEpisodes" of the game.
    After each episode, the returns of each action taken are calculated and used to update the Q-values.
    The new Q-values are used to play the next episode and so on.

    Parameters
    ----------
    env : gym.envs
        The OpenAI Gym environment used to play the game

    epsilon : float
        0 <= epsilon <= 1
        Epsilon value used in the Epsilon-Greedy policy when choosing an action
        Percentage of the time to take a random action

    gamma : float
        0 <= gamma <= 1
        The discount factor of future rewards encountered during an episode

    alpha : float
        0 <= alpha <= 1
        Learning rate when updating Q

    numEpisodes : int
        Number of episodes to play

    initialQ : defaultdict(tuple: numpy.ndarray)
        Q-value dictionary to base the policy of the game on (uses the Epsilon-Greedy policy)
        A copy is made of "initialQ" and the Q-values of each action taken during an episode are updated after the episode concludes
        The updated Q-value dictionary is then used to play the next episode

        Structure of dictionary:
            key : tuple
                state of game
            value : numpy.ndarray
                1D array of type float
                array of possible actions in state, indexed by action

        "initialQ" is a defaultdict that creates an array of actions if state hasn't yet been visited

    Returns
    -------
    defaultdict(tuple: numpy.ndarray)
        "initialQ" after being updated using the states, actions and returns of each episode
    """
    Q = deepcopy(initialQ)

    runningAvgWindowSize = numEpisodes/10
    totalReturns = np.empty(numEpisodes)

    for n in range(numEpisodes):
        if n % 1000 == 0:
            print(n)

        statesActionsReturns = playEpisode(env, epsilon, gamma, Q)

        totalReturn = sum([sar[2] for sar in statesActionsReturns])
        totalReturns[n] = totalReturn

        Q = updateQValues(Q, alpha, statesActionsReturns)

    plotRunningAverage(totalReturns, runningAvgWindowSize)

    return Q


def flattenQDictToDataframe(qDict):
    """
    Converts a dictionary of Q-values to a pandas dataframe for easy postprocessing and saving to file

    Parameters
    ----------
    qDict : dict(tuple: numpy.ndarray)
        Q-value dictionary to convert

        Structure of dictionary:
            key : tuple
                state of game
            value : numpy.ndarray
                1D array of type float
                array of possible actions in state, indexed by action

    Returns
    -------
    pandas.DataFrame
        "qDict" after converted into a dataframe
    """
    df = pd.DataFrame.from_dict(qDict).T
    df.index.names = ["player", "dealer", "ace"]
    df.columns = ["action: stand", "action: hit"]

    return df


if __name__ == '__main__':
    """
    state = (score: int, dealer_score: int, ace: boolean) where 1 <= score, dealer_score <= 21
    total number of states = 21 x 21 x 2 = 882
    in this implementation there are exactly 2 actions for every state in blackjack:
        actions: 0 = stand, 1 = hit
    total number of state-action pairs, Q(s,a) = 882 x 2 = 1,764
    """
    env = gym.make("Blackjack-v0")

    numEpisodes = 10000
    epsilon = 0.1
    gamma = 0.9
    alpha = 0.1

    # because we know action space is consistent, can just create blank actions for a new state when encountered
    initialQ = defaultdict(lambda: np.random.rand(env.action_space.n) * 0.0001)

    Q = monteCarlo(env, epsilon, gamma, alpha, numEpisodes, initialQ)

    df = flattenQDictToDataframe(Q)
    df.to_csv("qTable.csv", sep=",")

