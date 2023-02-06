import json

from flask import Flask
from sklearn import preprocessing as p

from src.constants import DATA_FILE_PATHS, INPUT_KEYS
from src.utils import load_data, sample_data, format_data
from src.vae import VAE

app = Flask(__name__)

@app.route('/')
def index():
  SAMPLE_COUNT = 8

  vae = VAE.load('model')
  scaler = p.MinMaxScaler()
  data = load_data(DATA_FILE_PATHS, INPUT_KEYS)
  samples = sample_data(data, SAMPLE_COUNT)
  x_train = scaler.fit_transform(samples)
  reconstructed_data, _ = vae.reconstruct(x_train)
  denormalized_data = scaler.inverse_transform(reconstructed_data)
  response = format_data(denormalized_data, INPUT_KEYS)

  return json.dumps(response)

app.run()
