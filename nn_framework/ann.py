import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")

class ANN(object):
  def __init__(
    self,
    model          = None,
    expected_range = (-1, 1),
  ):
    self.layers = model
    self.error_history = []
    self.n_iter_train = int(1e8)
    self.n_iter_evaluate = int(1e6)
    self.viz_interval = int(1e5)
    self.reporting_bin_size = int(1e3)
    self.report_min = -3
    self.report_max = 0

    self.expected_range = expected_range
    
    self.reports_path = "reports"
    self.report_name  = "performance_history.png"
    
    try:
      os.mkdir("reports")
    except Exception:
      print("Can't create reports folder")
  
  def train(self, training_set):
    for i_iter in range(self.n_iter_train):
      x = next(training_set()).ravel()
      y = self.forward_prop(x)
      self.error_history.append(1)
      
      if (i_iter + 1) % self.viz_interval == 0:
        self.report
    
  
  def evaluate(self, evaluation_set):
    for i_iter in range(self.n_iter_evaluate):
      x = next(evaluation_set()).ravel()
      y = self.forward_prop(x)
      self.error_history.append(1)
      
      if (i_iter + 1) % self.viz_interval == 0:
        self.report
      
  def normalize(self, values):
    """
    Transform the input/output values so that they tend to
    fall between -5 and 5
    """
    min_val = self.expected_range[0]
    max_val = self.expected_range[1]
    scale_factor = max_val - min_val
    offset_factor = min_val
    return (values - offset_factor) / scale_factor - .5
  
  def denormalize(self, transformed_values):
    min_val = self.expected_range[0]
    max_val = self.expected_range[1]
    scale_factor = 2 / (max_val - min_val)
    offset_factor = min_val - 1
    return transformed_values / scale_factor - offset_factor
  
  def forward_prop(self, x):
    y = x.ravel()[np.newaxis, :]
    
    for layer in self.layers:
      y = layer.forward_prop(y)
    return y.ravel()
  
  def backward_prop(self):
    pass

  def report(self):
    n_bins = int(len(self.error_history) // self.reporting_bin_size)
    smoothed_history = []
    for i_bin in range(n_bins):
      smoothed_history.append(np.mean(self.error_history[
                                      i_bin * self.reporting_bin_size:
                                      (i_bin + 1) * self.reporting_bin_size
                                      ]))
    error_history = np.log10(np.array(smoothed_history) + 1e-10)
    ymin = np.minimum(self.report_min, np.min(error_history))
    ymax = np.maximum(self.report_max, np.max(error_history))
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(error_history)
    ax.set_xlabel(f"x{self.reporting_bin_size} iterations")
    ax.set_ylabel("log error")
    ax.set_ylim(ymin, ymax)
    ax.grid()
    fig.savefig(os.path.join(self.reports_path, self.report_name))
    plt.close()