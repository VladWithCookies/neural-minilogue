import json
import numpy as np

def load_data(paths, keys):
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

def to_patch(array, keys):
  result = {}

  for index in range(len(array)):
    key = keys[index]
    value = array[index]
    result[key] = int(value)

  return result

def format_data(data, keys):
  result = []

  for array in data:
    patch = to_patch(array, keys)
    result.append(patch)

  return result
