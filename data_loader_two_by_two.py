import numpy as np


def get_data_sets():
  toy_dataset = [
    np.array([
      [0, 0],
      [1, 1]
    ]),
    np.array([
      [1, 0],
      [1, 0]
    ]),
    np.array([
      [1, 1],
      [0, 0]
    ]),
    np.array([
      [0, 1],
      [0, 1]
    ]),
    np.array([
      [1, 0],
      [0, 1]
    ]),
    np.array([
      [1, 1],
      [0, 1]
    ]),
    np.array([
      [0, 1],
      [0, 0]
    ]),
    np.array([
      [0, 1],
      [1, 0]
    ])]
  
  def training_set():
    while True:
      index = np.random.choice(len(toy_dataset))
      yield toy_dataset[index]
      
  def evaluation_set():
    while True:
      index = np.random.choice(len(toy_dataset))
      yield toy_dataset[index]
      
  return training_set, evaluation_set
