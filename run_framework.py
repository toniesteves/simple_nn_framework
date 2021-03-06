import data_loader_two_by_two as dat
from nn_framework.ann import ANN
from nn_framework.layer import Dense
from nn_framework.activation import tahn

N_NODES = [7, 4, 6]

train_set, eval_set = dat.get_data_sets()

sample = next(train_set())
input_value_range = (-5, 5)
n_pixels = sample.shape[0] * sample.shape[1]

n_nodes = [n_pixels] + N_NODES + [n_pixels]

model = []

# print("Hidden Layers")
# print(N_NODES)
#
# print("Network Archteture")
# print(n_nodes)

for idx_layer in range(len(n_nodes) - 1):
    # print(" From " + str(n_nodes[idx_layer]) + " neurons to " + str(n_nodes[idx_layer + 1]) + " neurons.")
    # print("--")
    model.append(Dense(
      n_nodes[idx_layer],
      n_nodes[idx_layer + 1],
      tahn))

autoencoder = ANN(
  model          = model,
  expected_range = input_value_range)
autoencoder.train(train_set)
autoencoder.evaluate(eval_set)
