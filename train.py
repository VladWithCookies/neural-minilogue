from sklearn import preprocessing as p

from src.constants import DATA_FILE_PATHS, INPUT_KEYS
from src.utils import load_training_data
from src.vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 1000

def train(x_train, learning_rate, batch_size, epochs):
  vae = VAE(
    input_shape = (39,),
    hidden_nodes = (16,),
    latent_nodes = 8
  )

  vae.summary()
  vae.compile(learning_rate)
  vae.train(x_train, batch_size, epochs)

  return vae

scaler = p.MinMaxScaler()
x_train = scaler.fit_transform(load_training_data(DATA_FILE_PATHS, INPUT_KEYS))
model = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
model.save("model")
