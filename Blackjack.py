import random
from copy import deepcopy
from enum import Enum
from functools import reduce

import numpy as np
import pandas as pd
import seaborn as sns
import gym
import tensorflow as tf

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import DeepQNetworkTensorflow
from DeepQNetworkTensorflow import *



# Load an existing model, or train a new one
class ModelInit(Enum):
    NEW  = 0
    LOAD = 1



'''
Get an action to take using the Epsilon-Greedy policy
Take a random action 'epsilon' percent of the time
Otherwise take the best action (the action with the highest value)

Parameters
----------
epsilon : float
    0 <= epsilon <= 1
    Percentage of the time to take a random action in actionValues

actionValues : numpy.ndarray
    1D array of type float
    The possible actions to take given a state, where the array is indexed by
    action

Returns
-------
int
    The index of the action to take
'''
def epsilonGreedyPolicy(epsilon, actionValues):
    p = np.random.random()

    return np.argmax(actionValues) if p < (1 - epsilon) else random.randint(0, len(actionValues) - 1)


'''
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
    Q-value dictionary to base the policy of the game on
    (uses the Epsilon-Greedy policy)

    Structure of dictionary:
        key : tuple
            state of game (observation from env)
        value : numpy.ndarray
            1D array of type float
            array of possible actions in state, indexed by action

    Q is a defaultdict that creates an array of actions if state hasn't yet
    been visited

Returns
-------
array-like(tuple, int, float)
    Array-like object of triples: (states, actions, rewards) from game
    Be aware of the timing:
        each triple is s(t), a(t), r(t)
        but r(t) results from taking action a(t-1) from s(t-1)
        and then landing in s(t)
'''
def getStatesActionsRewardsFromPlayingGame(env, epsilon, Q):
    done = False

    s = env.reset()
    a = epsilonGreedyPolicy(epsilon, Q[s])
    statesActionsRewards = [(s, a, 0)]

    while not done:
        s, r, done, info = env.step(a)

        if done:
            statesActionsRewards.append((s, None, r))
            break
        else:
            a = epsilonGreedyPolicy(epsilon, Q[s])
            statesActionsRewards.append((s, a, r))

    return statesActionsRewards


'''
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

statesActionsRewards : array-like(tuple, int, float)
    Array-like object of triples: (states, actions, rewards) to calculate
    returns from

Returns
-------
array-like(tuple, int, float)
    Array-like object of states, actions, returns from game
'''
def calculateReturns(gamma, statesActionsRewards):
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


'''
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
    Q-value dictionary to base the policy of the game on
    (uses the Epsilon-Greedy policy)

    Structure of dictionary:
        key : tuple
            state of game (observation from env)
        value : numpy.ndarray
            1D array of type float
            array of possible actions in state, indexed by action

    Q is a defaultdict that creates an array of actions if state hasn't yet
    been visited

Returns
-------
array-like(tuple, int, float)
    Array-like object of states, actions, returns from game
'''
def playEpisode(env, epsilon, gamma, Q):
    statesActionsRewards = getStatesActionsRewardsFromPlayingGame(env, epsilon, Q)

    return calculateReturns(gamma, statesActionsRewards)


'''
Updates Q-values of 'Q' using the states, actions and returns from an episode
Update is according to the following formula:
    Q[s][a] <- Q[s][a] + alpha * (G - Q[s][a])

Note
----
This function mutates Q!
Can't use a functional approach because making copies of Q here is very slow
when calling this function multiple times

Parameters
----------
Q : defaultdict(tuple: numpy.ndarray)
    Q-value dictionary to update using the states, actions and returns from an
    episode

    Structure of dictionary:
        key : tuple
            state of game
        value : numpy.ndarray
            1D array of type float
            array of possible actions in state, indexed by action

    Q is a defaultdict that creates an array of actions if state hasn't yet
    been visited

alpha : float
    0 <= alpha <= 1
    Learning rate when updating Q

statesActionsReturns : array-like(tuple, int, float)
    Array-like object of states, actions, returns from an episode used to update
    Q-values in Q

Returns
-------
defaultdict(tuple: numpy.ndarray)
    Updated version of Q using the states, actions and returns of an episode
'''
def updateQValues(Q, alpha, statesActionsReturns):
    seenStateActionPairs = set()

    for s, a, G in statesActionsReturns:
        sa = (s, a)
        if sa not in seenStateActionPairs:
            Q[s][a] = Q[s][a] + alpha * (G - Q[s][a])

            seenStateActionPairs.add(sa)

    return Q


'''
Helper function to map from an iterable of layer sizes to a 'x-delimimted'
string representation.
E.g.
    layerSizesToString([1, 2, 3]) -> '1x2x3'

Parameters
----------
layerSizes : array-like(int)
    Iterable containing the layer sizes

Returns
-------
string
    x-delimimted string representation of the layer sizes
'''
def layerSizesToString(layerSizes):
    return reduce(lambda acc, x: f'{acc}x{x}', layerSizes[1:], layerSizes[0])


'''
Plots the running average of the total returns of the episodes against the
number of episodes run

Parameters
----------
totalReturns : array-like
    1D array of floats
    Array-like structure of the total returns of each episode

windowSize : int
    The size of the window of the running average
'''
def plotRunningAverage(totalReturns, windowSize, hiddenLayerSizes):
    # Reduce the number of points used in the plot
    numberOfPointsToDisplay = 10000

    N = len(totalReturns)
    runningAverage = np.empty(N)

    for n in range(N):
        runningAverage[n] = totalReturns[
                                max(0, n - int(windowSize)):(n + 1)
                            ].mean()

    # make sure number of points to display <= number of games played
    if numberOfPointsToDisplay > N:
        numberOfPointsToDisplay = N

    stepSize = int(N/numberOfPointsToDisplay)
    numberOfIterations = np.arange(0, N, stepSize)
    runningAverageDisplay = runningAverage[0::stepSize]

    mean = runningAverage.mean()
    std = runningAverage.std()
    figureSize = (8, 6)
    xLabel = 'Number of Games Played'
    yLabel = 'Running Average of Score'

    layerSizesString = layerSizesToString(hiddenLayerSizes)

    title = f'Number of Games vs Average Score\n'\
            f'Hidden Layer Sizes: {layerSizesString}\n'\
            f'(mean: {mean:.2f}, std: {std:.2f})'

    plt.figure('Linear Plot', figsize=figureSize)
    plt.plot(numberOfIterations, runningAverageDisplay)
    plt.xlabel(xLabel)
    plt.xlim(1, N)
    plt.ylabel(yLabel)
    plt.ylim(-1, 0)
    plt.title(title)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(f'Results-Linear-{layerSizesString}-{N}.png',
                bbox_inches='tight')

    plt.figure('Logarithmic Plot', figsize=figureSize)
    plt.semilogx(numberOfIterations, runningAverageDisplay, nonposx='clip')
    plt.xlabel(xLabel)
    plt.xlim(1, N)
    plt.ylabel(yLabel)
    plt.ylim(-1, 0)
    plt.title(title)
    plt.savefig(f'Results-Logarithmic-{layerSizesString}-{N}.png',
                bbox_inches='tight')


'''
Starting with the Q-values defined by initialQ, plays through numEpisodes
of the game.
After each episode, the returns of each action taken are calculated and used to
update the Q-values.
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
    Q-value dictionary to base the policy of the game on
    (uses the Epsilon-Greedy policy)
    A copy is made of initialQ and the Q-values of each action taken during an
    episode are updated after the episode concludes
    The updated Q-value dictionary is then used to play the next episode

    Structure of dictionary:
        key : tuple
            state of game
        value : numpy.ndarray
            1D array of type float
            array of possible actions in state, indexed by action

    initialQ is a defaultdict that creates an array of actions if state hasn't
    yet been visited

Returns
-------
defaultdict(tuple: numpy.ndarray)
    initialQ after being updated using the states, actions and returns of each
    episode
'''
def monteCarlo(env, epsilon, gamma, alpha, numEpisodes, initialQ):
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

    plotRunningAverage(totalReturns, runningAvgWindowSize, )

    return Q


'''
Converts a dictionary of Q-values to a pandas dataframe for easy postprocessing
and saving to file

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
    qDict after converted into a dataframe
'''
def flattenQDictToDataframe(qDict):
    df = pd.DataFrame.from_dict(qDict).T
    df.index.names = ['player', 'dealer', 'ace']
    df.columns = ['action: stand', 'action: hit']

    return df


# ------------------------- End of Classical RL Code ---------------------------


'''
Play one round (episode) of a game and train the Deep Q Networks (DQN) on the
rewards found during the episode.
Uses the 'Double Deep Q-Network' strategy for training.
i.e. decoupling action selection and action evaluation to stabilise training and
potentially speed up convergence.

while game not finished
    get starting state/observation
    sample action and get reward and next state

    add experience (s, a, r, s') to the experience replay buffer
    Compute one iteration of gradient descent, using targe model to
    predict the Q-values of the next state

    if hyperparameter 'copyPeriod' iterations have occurred
        copy targe model into model

Note: The weights of the networks are mutated in this function during training.

Parameters
----------
env : TimeLimit
    OpenAI Gym environment to use

model : DQN
    Deep Q-Network responsible for selection of the next action.

targetModel : DQN
    The Target Network.
    Used to calculate the targets used in training.
    Responsible for the evaluation of each action.

epsilon : float
    0 <= epsilon <= 1
    Epsilon value used in the Epsilon-Greedy policy when choosing an action
    Percentage of the time to take a random action

updatePeriod : int
    Copy weights from model to target model every updatePeriod steps of the game

Returns
-------
int
    Sum of the rewards encountered during the episode
'''
def playOne(env, model, targetModel, epsilon, updatePeriod):
    observation = env.reset()
    done = False
    totalReward = 0
    i = 0

    while not done:
        action = model.sampleAction(observation, epsilon)
        prevObservation = observation
        observation, reward, done, info = env.step(action)

        totalReward += reward

        # Update the model
        model.addExperience(prevObservation, action, reward, observation, done)
        model.train(targetModel)

        i += 1

        if i % updatePeriod == 0:
            targetModel.copyFrom(model)

    return totalReward


'''
Uses the 'Double Deep Q-Network' strategy for training.
i.e. decoupling action selection and action evaluation to stabilise training and
potentially speed up convergence.

Calls playOne() for 'numEpisodes' iterations, which plays one game then trains
Uses a decaying epsilon for exploration, see hyperparameters 'settledTimestep'
and 'finalEpsilon'

Note: The actual mutation of NN weights are changed during call to playOne()

Parameters
----------
env : TimeLimit
    OpenAI Gym environment to use

hp : Hyperparameters
    Dictionary containing the hyperparameters of the neural network

checkpointDir : string
    Path to directory containing TensorFlow checkpoint

modelInitialisation : ModelInit(Enum)
    Enum specifying whether to start with new weights or load saved weights for
    the neural network model

debugInfoPeriod : int
    How often to print stats while training

saveName : string
    Name for saved model
'''
def trainModel(env, hp, checkpointDir, modelInitialisation, debugInfoPeriod, saveName):
    model = DQN(env, hp)
    targetModel = DQN(env, hp)

    saver = tf.train.Saver(max_to_keep=4)

    with tf.Session() as sess:
        if (modelInitialisation == ModelInit.NEW):
            print('Initialising new model')
            init = tf.global_variables_initializer()
            sess.run(init)
        elif (modelInitialisation == ModelInit.LOAD):
            print('Loading model')
            ckpt = tf.train.latest_checkpoint(checkpointDir)
            saver.restore(sess, save_path=ckpt)

        model.setSession(sess)
        targetModel.setSession(sess)

        totalRewards = np.empty(hp['numEpisodes'])

        print('Begin training')
        for n in range(hp['numEpisodes']):
            if n < hp['settledTimestep']:
                m = (hp['finalEpsilon'] - 1)/(hp['settledTimestep'] - 0)
                c = 1
                epsilon = m*n + c
            else:
                epsilon = hp['finalEpsilon']

            reward = playOne(env, model, targetModel, epsilon, hp['copyPeriod'])
            totalRewards[n] = reward

            if n % debugInfoPeriod == 0 and n != 0:
                movingAverage = totalRewards[max(0, n - debugInfoPeriod):(n + 1)].mean()
                print(
                    f'episode: {n:3.3e},'
                    f' last reward: {reward:2.1f},'
                    f' epsilon: {epsilon:.3f},'
                    f' avg reward (last {debugInfoPeriod}): {movingAverage:.3f}'
                )

        print('Saving...')
        saver.save(sess, saveName)
        print('Save complete!')

        plotRunningAverage(totalRewards, 1000, hp['hiddenLayerSizes'])


'''
Restores a TensorFlow model and returns a session object

Parameters
----------
checkpointDir : string
    Path to directory containing TensorFlow checkpoint

Returns
-------
tensorflow.Session
    TensorFlow Session with restored model
'''
def createSession(checkpointDir):
    saver = tf.train.Saver(max_to_keep=4)

    sess = tf.InteractiveSession()
    ckpt = tf.train.latest_checkpoint(checkpointDir)

    saver.restore(sess, save_path=ckpt)

    return sess


'''
Helper to generate game states.
Used when generating CSV file of the learned action for each state

Returns
-------
array-like(tuple(int, int, boolean))
    Collection of game states where a game state is a 3-tuple of
        (player score, dealer score, player has usable ace)
'''
def createAllGameStates():
    start = 1
    # We have to + 1 because range() isn't inclusive
    end = 21 + 1

    playerValues = list(range(start, end))
    dealerValues = list(range(start, end))
    usableAces = [ False, True ]

    return [ (p, d, a)
             for a in usableAces
             for d in dealerValues
             for p in playerValues
           ]


'''
Loads trained DQN and saves a CSV file of the learned action for each game state

Parameters
----------
env : TimeLimit
    OpenAI Gym environment to construct the deep Q-learning network

hp : Hyperparameters
    Dictionary containing hyperparameters to construct the deep Q-learning network

checkpointDir : string
    Path to directory containing TensorFlow checkpoint
'''
def saveModelDecisionMatrix(env, hp, checkpointDir):
    states = createAllGameStates()

    # Reset the graph in case one already exists when this function is called
    tf.reset_default_graph()

    # No need to set targetModel
    model = DQN(env, hp)
    # targetModel = DQN(env, hp)

    sess = createSession(checkpointDir)

    model.setSession(sess)
    # targetModel.setSession(sess)

    results = list(map(
        lambda s: (*s, model.sampleAction(s)),
        states))

    df = pd.DataFrame(results, columns=['player', 'dealer', 'ace', 'action'])

    df.to_csv(deepQLearningCsvName, sep=',', index=False)
    print(f'Saved results as {deepQLearningCsvName}')


'''
Takes a Pandas DataFrame with the columns ['player', 'dealer', 'ace', 'action']
 and creates two separate heatmaps of ['player', 'dealer', 'action'] for each
'ace' state
('ace' is a boolean)

X-Axis: Dealer Score
Y-Axis: Player Score
Z-Axis: Learned Action (0: sit, 1: hit)

Parameters
----------
df : pandas.DataFrame
    df has the columns ['player', 'dealer', 'ace', 'action']
        types: [int, int, boolean, int]

titlePrefix : string
    The title of the saved figure
    The 'ace' state will be appended to this title

filenamePrefix : string
    Filename to save as
    One heatmap will be saved for each 'ace' state
'''
def plotDataframeAsHeatmap(df, titlePrefix, filenamePrefix):
    aceValues = [False, True]

    def saveHeatmap(ace):
        filteredData = df[df['ace'] == ace] \
                         .pivot(index='player',
                                columns='dealer',
                                values='action')

        plt.clf()

        ax = sns.heatmap(filteredData,
                         cmap='viridis',
                         cbar_kws={'label': 'action (0: stand, 1: hit)'},
                         linewidths=.5) \
                .set_title(f'{titlePrefix} for ace={str(ace)}')

        plt.gca().invert_yaxis()

        figure = ax.get_figure()
        figure.savefig(f'{filenamePrefix}Ace={str(ace)}.png',
                       bbox_inches='tight')

    list(map(saveHeatmap, aceValues))


'''
Plot heatmaps for the classical Q-learning and the Deep Q-Network data

Parameters
----------
qLearningCsvName : string
    Path to the classical Q-learning CSV file

deepQLearningCsvName : string
    Path to the deep Q-learning CSV file
'''
def plotHeatmaps(qLearningCsvName, deepQLearningCsvName):
    print('Reading CSV files')
    qLearningDf = pd.read_csv(qLearningCsvName)
    deepQLearningDf = pd.read_csv(deepQLearningCsvName)

    sortCols = ['ace', 'player', 'dealer']

    def qValuesToAction(row):
        if row['action: stand'] > row['action: hit']:
            return 0
        else:
            return 1

    print('Transforming data...')
    qLearningDf['action'] = qLearningDf.apply(qValuesToAction, axis=1)
    qLearningDf = qLearningDf.drop('action: stand', axis=1)
    qLearningDf = qLearningDf.drop('action: hit', axis=1)
    qLearningDf = qLearningDf.sort_values(by=sortCols)

    deepQLearningDf = deepQLearningDf.sort_values(by=sortCols)

    joinCols = ['player', 'dealer', 'ace']
    combinedTable = qLearningDf.merge(deepQLearningDf,
                                      left_on=joinCols,
                                      right_on=joinCols,
                                      suffixes=(': Q-Table', ': Deep-Q NN'))
    # combinedTable.to_csv('combinedTable.csv', sep=',', index=False)

    # The deep Q-learning network's data contains unreachable game states
    # i.e. 1 <= dealer <= 21 and 1 <= player <= 21
    # I purposely kept impossible states in the deep learning data because
    # it may help to identify trends in behaviour that the neural network
    # developed and for curiosity sake
    # Crop the data to only the states reachable in-game.
    # E.g. a state with player = 1 is impossible because the player must draw
    # two cards at the start (it wouldn't make sense to sit with a single card)
    # Achievable scores:
    #   1  <= dealer <= 10
    #   4  <= player <= 21 with NO ace
    #   12 <= player <= 21 with ace
    # As the data already has the upper and lower bounds of 21 and 1
    # respectively, no need to include them in the constraints here
    # dealer <= 10 & ((player <= 4 & ace == False) | (player <= 10 & ace == True))
    deepQLearningCropped = deepQLearningDf[
        (deepQLearningDf['dealer'] <= 10)
        & (
             (deepQLearningDf['player'] >= 4 ) & (deepQLearningDf['ace'] == False)
           | (deepQLearningDf['player'] >= 12) & (deepQLearningDf['ace'] == True)
        )
    ]

    print('Plotting heatmaps...')
    plotDataframeAsHeatmap(qLearningDf,
                           'Q-Learning: Learned Strategy',
                           'qLearning')
    plotDataframeAsHeatmap(deepQLearningCropped,
                           'Deep Q-Learning: Learned Strategy',
                           'deepQLearningCropped')
    plotDataframeAsHeatmap(deepQLearningDf,
                           'Deep Q-Learning: Learned Strategy',
                           'deepQLearning')

    print('Completed plotting heatmaps')


if __name__ == '__main__':
    '''
    state = (score: int, dealer_score: int, ace: boolean) where
        Valid game states:
            Ace = False: 4 <= player <= 21, 1 <= dealer <= 10
                18 x 10
            Ace = True: 12 <= player <= 21, 1 <= dealer <= 10
                10 x 10
    See the filtering of data in plotHeatmaps() for more detail
    Total valid game states = 280
    In this implementation there are exactly 2 actions for every state in blackjack:
        actions: 0 = stand, 1 = hit
    Total number of state-action pairs, Q(s,a) = 280 x 2 = 560
    '''

    env = gym.make('Blackjack-v0')

    hp = Hyperparameters(numEpisodes=int(1.1e7),
                         copyPeriod=100,
                         gamma=0.9,
                         alpha=1e-3,
                         batchSize=1000,
                         minExperiences=10000,
                         maxExperiences=100000,
                         settledTimestep=7e6,
                         finalEpsilon=0.1,
                         hiddenLayerSizes=[10, 10, 5])

    # -------------------------------- Settings --------------------------------

    # Specify if saving or loading a model
    modelInitialisation = ModelInit.NEW
    # Restoring/saving model name
    saveName = 'myTrainedModel'
    checkpointDir = './'

    # How often to print debug info
    debugInfoPeriod = int(1e4)

    # Saving csv files
    qLearningCsvName = 'qTable.csv'
    deepQLearningCsvName = 'deepQLearningResults.csv'


    trainModel(env,
               hp,
               checkpointDir,
               modelInitialisation,
               debugInfoPeriod,
               saveName)
    saveModelDecisionMatrix(env, hp, checkpointDir)
    plotHeatmaps(qLearningCsvName, deepQLearningCsvName)
