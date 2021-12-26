import gym
import os
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam, RMSprop


def Model(inputShape, actionSpace):
    xInput = Input(inputShape)

    # Input layer
    X = Dense(512, input_shape=inputShape, activation="relu",
              kernel_initializer='he_uniform')(xInput)

    # Hidden layer
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output layer
    X = Dense(actionSpace, activation="linear",
              kernel_initializer='he_uniform')(X)

    model = Model(inputs=xInput, outputs=X, name='CartPoleV1Model')
    model.compile(loss="mse", optimizer=RMSprop(
        lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.summary()

    return model


class Agent:

    def __init__(self):

        # Use the OpenAI Gym CartPole-v1 environment
        self.environment = gym.make("CartPole-v1")

        # Get the state and action size
        self.stateSize = self.env.observation_space.shape[0]
        self.actionSize = self.env.action_space.n

        # Number of episodes the agent will play
        self.EPISODES = 1000

        # If memory is exceeded, popleft()
        self.memory = deque(maxlen=2000)

        # Discount rate
        self.gamma = 0.95

        # Exploration rate
        self.epsilon = 1.0

        # Minimum epsilon
        self.epsilonMin = 0.001

        # Decrease the number of explorations as the agent gets better at the game
        self.epsilonDecay = 0.999

        # How much memory the DQN will use to learn
        self.batchSize = 64

        # Minimum size of memory to start training
        self.minMemory = 1000

        # Initialize the model
        self.model = Model(input_shape=(self.stateSize,),
                           action_space=self.actionSize)

    def remember(self, state, action, reward, nextState, gameOver):

        # If memory is exceeded, popleft()
        self.memory.append((state, action, reward, nextState, gameOver))

        # If memory is greater than the minimum size of memory to start training
        if len(self.memory) > self.minMemory:

            # If epsilon is greater than the minimum epsilon
            if self.epsilon > self.epsilonMin:

                # Decay epsilon
                self.epsilon *= self.epsilonDecay

    def reply(self):

        # If memory is less than the minimum size of memory to start training
        if len(self.memory) < self.minMemory:
            return

        # Sample a random batch from memory or all of memory
        batch = random.sample(self.memory, min(
            len(self.memory), self.batchSize))

        # Initialize state, action, reward, nextState, and gameOver lists
        state = np.zeros((self.batchSize, self.stateSize))
        action = []
        reward = []
        nextState = np.zeros((self.batchSize, self.stateSize))
        gameOver = []

        # Store each entry of the batch in the lists
        for i in range(self.batchSize):
            state[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            nextState[i] = batch[i][3]
            gameOver.append(batch[i][4])

        # Predict
        target = self.model.predict(state)
        nextTarget = self.model.predict(nextState)

        for i in range(self.batchSize):

            # Q value for the action
            if gameOver[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + \
                    self.gamma * (np.amax(nextTarget[i]))

        # Train
        self.model.fit(state, target, batch_size=self.batchSize, verbose=0)

    def train(self):

        # Loop through every episode the agent will play
        for i in range(self.EPISODES):

            # Reset the state and gameOver
            state = self.environment.reset()
            state = np.reshape(state, [1, self.stateSize])
            gameOver = False

            # Initialize a counter
            count = 0

            # While the game is not over
            while not gameOver:

                # Display the environment
                self.environment.render()

                # If a random float between 0 and 1 is less than or equal to epsilon
                if np.random.random() <= self.epsilon:

                    # Perform a random action
                    action = random.randrange(self.actionSize)

                # If a random float between 0 and 1 is greater than epsilon
                else:

                    # Perform the predicted action
                    action = np.argmax(self.model.predict(state))

                # Pull nextState, reward, gameOver from the action
                nextState, reward, gameOver, _ = self.environment.step(action)
                nextState = np.reshape(nextState, [1, self.stateSize])

                # If the game is not over
                if not gameOver or count == self.environment._max_episode_steps - 1:

                    # Keep the reward as is
                    reward = reward

                # If the game is over
                else:

                    # Change the reward to -100
                    reward = -100

                # remember the game iteration
                self.remember(state, action, reward, nextState, gameOver)

                # Set the state as the next state
                state = nextState

                # Increment the counter
                count += 1

                # If the game is over print the game statistics
                if gameOver:
                    print("Episode: {}/{}, Score: {}, Epsilon: {:.2}".format(i,
                                                                             self.EPISODES, count, self.epsilon))

                    # If maximum score
                    if count == 500:

                        # Save the model
                        print("Saving trained model as CartPole-V1.h5")
                        self.model.save("CartPole-V1.h5")

                        return

                # Else replay the game
                self.replay()
