import numpy as np
import tensorflow as tf


def Hyperparameters(numEpisodes=int(5e6),
                    copyPeriod=100,
                    gamma=0.9,
                    alpha=1e-3,
                    batchSize=1000,
                    minExperiences=10000,
                    maxExperiences=100000,
                    settledTimestep=3e6,
                    finalEpsilon=0.1,
                    hiddenLayerSizes=[10, 10, 5]):
    return dict(numEpisodes=numEpisodes,
                # How often to sync target model with model
                copyPeriod=copyPeriod,
                # Discount factor
                gamma=gamma,
                # Learning rate
                alpha=alpha,

                # Experience replay buffer parameters
                # Number of samples to train on during training step
                batchSize=batchSize,
                minExperiences=minExperiences,
                maxExperiences=maxExperiences,

                # Epsilon-Greedy decay parameters
                # Epsilon starts at 1 then decays linearly until 'settledTimestep' episodes
                # and then remains at constant value 'finalEpsilon'
                settledTimestep=settledTimestep,
                finalEpsilon=finalEpsilon,

                hiddenLayerSizes=hiddenLayerSizes)


'''
Layer for a neural network, keeps track of its weights

Properties
----------
W : tensorflow.Variable(shape = inputLayerSize x outputLayerSize)
    The weights of the hidden layer
    Initialised with random values between 0 and 1

params : array-like(tensorflow.Variable)
    Array-like structure holding the weights, required in order to copy a network

activationFunc : function(tensorflow.Variable)
    Activation function that takes a Tensorflow tensor as an argument

useBias : boolean
    Flag that determines whether to use a bias term in the weights of the layer

b : tensorflow.Variable(shape = 1 x outputLayerSize)
    The bias term of the layer
'''
class HiddenLayer:
    def __init__(self,
                 inputLayerSize,
                 outputLayerSize,
                 activationFunc=tf.nn.tanh,
                 useBias=True):

        self.W = tf.Variable(
                    tf.random_normal(shape=(inputLayerSize, outputLayerSize)))
        self.params = [self.W]
        self.useBias = useBias

        if useBias:
            self.b = tf.Variable(np.zeros(outputLayerSize).astype(np.float32))
            self.params.append(self.b)

        self.activationFunc = activationFunc

    '''
    Forward the input tensor X through the layer
    Computes the result of (matrix algebra)
        f(X * W + b)
    where
        f = activation function
        X = input values
        W = weights of hidden layer
        b = bias tensor (if applicable)

    Parameters
    ----------
    X : tensorflow.Variable
        Input parameters into the layer

    Returns
    -------
    tensorflow.Variable
        f(X * W + b)
    '''
    def forward(self, X):
        if self.useBias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.activationFunc(a)




'''
Neural network class for Deep Q-learning

Key Parameters
----------------------
inputSpaceSize : int
    Size of the input tensor i.e. the observation space size

outputSpaceSize : int
    Size of the output tensor i.e. the action space size

hiddenLayerSizes : array-like(int)
    Array-like structure holding the sizes of the hidden layers

gamma : float
    0 <= gamma <= 1
    Discount factor used in training. See train().

alpha : float
    0 <= alpha <= 1
    Learning rate used by training optimiser.

minExperiences : int
    Minimum amount of states in the experience buffer required in order to train

maxExperiences : int
    Maximum size of the experience buffer

batchSize : int
    Amount of experiences to train on during the train() step
'''
class DQN:
    def __init__(self, env, hp):
        observationSpaceSize = len(env.observation_space.sample())
        actionSpaceSize = env.action_space.n

        self.outputSpaceSize = actionSpaceSize

        # create the graph
        self.layers = []
        M1 = observationSpaceSize
        for M2 in hp['hiddenLayerSizes']:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
        # final layer
        layer = HiddenLayer(M1, actionSpaceSize, lambda x: x)
        self.layers.append(layer)

        # collect params for copy
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # inputs
        self.X = tf.placeholder(tf.float32,
                                shape=(None, observationSpaceSize),
                                name='X')
        # targets
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.actionValues = Y_hat

        selectedActionValues = tf.reduce_sum(
            Y_hat * tf.one_hot(self.actions, actionSpaceSize),
            reduction_indices=[1]
        )

        cost = tf.reduce_sum(tf.square(self.G - selectedActionValues))
        self.trainingOptimiser = tf.train.AdamOptimizer(hp['alpha']).minimize(cost)
        # self.trainingOptimiser = tf.train.AdagradOptimizer(1e-2).minimize(cost)
        # self.trainingOptimiser = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
        # self.trainingOptimiser = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

        # create replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.maxExperiences = hp['maxExperiences']
        self.minExperiences = hp['minExperiences']
        self.batchSize = hp['batchSize']
        self.gamma = hp['gamma']


    # Move this into the constructor?
    def setSession(self, session):
        self.session = session


    '''
    Copy the weights from DQN 'other' to this instance

    Parameters
    ----------
    other : DQN
        Network's parameters to copy into this instance
    '''
    def copyFrom(self, other):
        # collect all the ops
        ops = []
        myParams = self.params
        otherParams = other.params

        # Compute the Tensorflow graph for 'other' network
        # Assign the newly computed weights to this network
        # Then compute the graph for this network
        for p, q in zip(myParams, otherParams):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)

        self.session.run(ops)


    '''
    Feed input tensor X through the graph and compute the output

    Parameters
    ----------
    X : tensorflow.Variable
        Input parameters into the neural network

    Returns
    -------
    array_like(float)
        Array-like of the Q-values of the actions for the current state, X
        Q-values are indexed by the actions in the action space
    '''
    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.actionValues, feed_dict={self.X: X})


    '''
    Computes one iteration of gradient descent using a random batch of
    experience from the experience replay buffer:

        Randomly select a batch of samples from experience replay buffer.
        Predict Q-values for taking each action in the next state and get the
        max Q-value of these actions.
            (i.e. get the Q-value of taking the best action, given the state)
        Calculate targets via
            target = reward + gamma * max Q-value of next state
            i.e. Q-learning formula

        Compute one operation of gradient descent with the batch of
        states, targets and actions.
            The optimiser will try and reduce the loss (TD error) where
                loss = (target - predicted value)^2
            ('target' is formally known as the 'TD-Target') 

    Note that although both target and predicted value involve a prediction of
    the Q-value from a neural network,
    the target is considered less biased as it includes the true reward observed
    from the game.
    This fact is what guarantees that gradient descent converges in deep
    Q-learning.
    i.e. the neural network's predictions should converge to the true rewards.

    Parameters
    ----------
    targetNetwork : DQN
        DQN used to predict the Q-values, and as a result of this,
        the targets of the next state
    '''
    def train(self, targetNetwork):
        if len(self.experience['s']) < self.minExperiences:
            return

        idx = np.random.choice(len(self.experience['s']),
                               size=self.batchSize,
                               replace=False)

        states     = [self.experience['s'][i]    for i in idx]
        actions    = [self.experience['a'][i]    for i in idx]
        rewards    = [self.experience['r'][i]    for i in idx]
        nextStates = [self.experience['s2'][i]   for i in idx]
        dones      = [self.experience['done'][i] for i in idx]

        nextQs = np.max(targetNetwork.predict(nextStates), axis=1)
        targets = [r + self.gamma * next_q if not done else r
                   for r, next_q, done in zip(rewards, nextQs, dones)]

        self.session.run(
            self.trainingOptimiser,
            feed_dict={
                self.X: states,
                self.G: targets,
                self.actions: actions
            }
        )


    '''
    Add experience to the experience replay buffer,
    Note experience replay buffer is a FIFO buffer of size maxExperiences

    Note the timing:
        s, a, r, s', done
    corresponds to
        prevObservation, action, reward, observation, done
    '''
    def addExperience(self, s, a, r, s2, done):
        if len(self.experience['s']) >= self.maxExperiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)


    '''
    Get an action using Epsilon-Greedy policy

    Parameters
    ----------
    observation : array_like(float)
        Array-like of the current observation (state)

    epsilon : float
        0 <= epsilon <= 1
        Epsilon value used in the Epsilon-Greedy policy when choosing an action
        Percentage of the time to take a random action

    Returns
    -------
        int
            The index of the action to take
    '''
    def sampleAction(self, observation, epsilon=0):
        if np.random.random() < epsilon:
            return np.random.choice(self.outputSpaceSize)
        else:
            X = np.atleast_2d(observation)
            return np.argmax(self.predict(X)[0])
