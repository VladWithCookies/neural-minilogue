import json
import numpy as np

def load_training_data(paths, keys):
  result = []

  for path in paths:
    with open(path) as file:
      data = json.load(file)

      for item in data:
        values = dict((key, item[key]) for key in keys if key in item).values()
        result.append(list(values))

  return np.array(result)

def sample_data(data, count):
  return data[np.random.choice(data.shape[0], count, replace = False), :]
