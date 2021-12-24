import gym
import random

environment = gym.make("CartPole-v1")


def game():

    # Each episode is its own game
    for episode in range(10):

        # Reset the environment
        environment.reset()

        # Maximum of 500 frames
        for i in range(500):

            # Display the environment
            environment.render()

            # Create a sample action in the environment, 0 OR 1 (left OR right)
            action = environment.action_space.sample()

            # Execute the environment with the action
            nextState, reward, gameOver, info = environment.step(action)

            # Print
            print(i, nextState, reward, gameOver, info, action)

            # Break the loop if the game is over
            if gameOver:
                break


game()
