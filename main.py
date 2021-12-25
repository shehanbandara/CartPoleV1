import gym
import os
import random
import numpy as np
from collections import deque
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam, RMSprop


def model(inputShape, actionSpace):
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
