import numpy as np
from nn_framework.activation import tahn

class Dense(object):
  def __init__(
    self,
    m_inputs,
    n_outputs,
    tahn,
    debug=False,
    
  ):
    self.debug = debug
    self.m_inputs = int(m_inputs)
    self.n_outputs = int(n_outputs)
    self.tahn = tahn
    
    self.learning_rate = .005
    
    self.initial_weights_scale = 1
    
    self.weights = self.initial_weights_scale * (np.random.sample(size=(self.m_inputs + 1, self.n_outputs)) * 2 - 1)
    
    self.w_grad = np.zeros((self.m_inputs + 1, self.n_outputs))
    
    self.x = np.zeros((1, self.m_inputs + 1))
    self.y = np.zeros((1, self.n_outputs))

  def forward_prop(self, inputs):
    """"
    Propagate the inputs foward through the network
    """
    bias = np.ones((1, 1))
    self.x = np.concatenate((inputs, bias), axis=1)
    v = self.x @ self.weights
    self.y = self.tahn.calc(v)
    return self.y
