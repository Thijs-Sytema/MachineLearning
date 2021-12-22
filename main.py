from _ml import MLAgent, train, save, load, train_and_plot, validate, plot_validation
from _core import is_winner, opponent, start
from _agent import RandomAgent
import random

class MyAgent(MLAgent):
    def evaluate(self, board):
        if is_winner(board, self.symbol):
            reward = 1
        elif is_winner(board, opponent[self.symbol]):
            reward = -1
        else:
            reward = 0
        return reward
    
    
random.seed(1)
 
my_agent = MyAgent()
random_agent = RandomAgent()

my_agent = MyAgent(alpha=0.5,epsilon=1.0)

train_and_plot(
    agent=my_agent,
    validation_agent=random_agent,
    iterations=50,
    trainings=100,
    validations=1000)