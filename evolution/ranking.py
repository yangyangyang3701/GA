import numpy as np


def ranking(self):
    # GA select the biggest one, but we want to minimize func, so we put a negative here
    self.FitV = -self.Y


def ranking_linear(self):
    self.FitV = np.argsort(np.argsort(-self.Y))
    return self.FitV
