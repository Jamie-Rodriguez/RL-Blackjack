# Artificial Intelligence - Reinforcement Learning: Learning to Play Blackjack
This is my reinforcement learning project; an agent that learns to play blackjack.

The agent learns by using a Monte Carlo strategy to sample returns and create a table of the _Q-values_ of the game, that is, the expected return of taking action _a_, given state _s_ - formalised as _Q(s, a)_.

A Monte Carlo strategy was chosen because it performs well on very episodic environments such as blackjack, where one episode (game) takes just three turns on average.

I'm using the _OpenAI Gym_ environment _Blackjack-v0_ as the environment for this project.

## Results
### Expected Return Using the Optimal Policy
After training for one million and then ten million games it can be found that the average score, given the optimal policy, converges to -0.12 i.e. playing blackjack under these specific rules gives a 0.12 advantage to the dealer/house in the long term.

Note: The environment does not allow insurance, splitting or doubling down on a hand.

![alt text](Results-Linear-1000000.png "Graph of the agent's performance while training - 1,000,000 samples")
![alt text](Results-Linear-10000000.png "Graph of the agent's performance while training - 10,000,000 samples")

### Learning Speed
I noticed that the agent very quickly learns to play with decent success.

After only ten thousand games, the agent approaches a value reasonably close to the true expected return (expected return from playing the optimal policy).

![alt text](Results-Linear-10000.png "Graph of the agent's performance while training - 10,000 samples")

Looking at the logarithmic plots shows that the agent converges to within 33.33% of the true value within 10<sup>4</sup> games, and within 5% after 10<sup>5</sup> games. 

![alt text](Results-Logarithmic-10000.png "Graph of the agent's performance while training - 10,000 samples")
![alt text](Results-Logarithmic-1000000.png "Graph of the agent's performance while training - 1,000,000 samples")
![alt text](Results-Logarithmic-10000000.png "Graph of the agent's performance while training - 10,000,000 samples")

## Extension to Model-Based Learning
By using information about the environment it is possible to achieve better performance.

Namely, if the agent can keep track of what cards have appeared in play i.e. "_card counting_", it can then make an even more informed decision. By modelling the state of the deck, greater accuracy can be achieved when predicting the probability of a card being drawn.

However, as I'm using the OpenAI Gym environment _Blackjack-v0_, the `draw_card` function simply generates a random number with no concept of a limited number of cards in the deck. See the source code below:

```python
def draw_card(np_random):
    return int(np_random.choice(deck))
```

In order to do model-based learning, an environment that includes a model of the state of the deck must first be created.

# To Do:
Replace the Q-Table with a neural network to act as a function approximator.

The objective of the neural network is to approximate function _Q(s, a)_, that maps from state, _s_, and action, _a_, to a value expressing how rewarding the choice of _a_ given _s_ is.

Given that the neural network behaves appropriately, we should expect to see the same results to those seen above.
