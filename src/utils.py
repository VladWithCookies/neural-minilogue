import json
import numpy as np

from sklearn import preprocessing as p

def load_training_data(paths, keys):
  result = []
  scaler = p.MinMaxScaler()

  for path in paths:
    with open(path) as file:
      data = json.load(file)

      for item in data:
        values = dict((key, item[key]) for key in keys if key in item).values()
        result.append(list(values))

  return np.array(scaler.fit_transform(result))
