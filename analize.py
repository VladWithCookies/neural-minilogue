from sklearn import preprocessing as p

from src.constants import DATA_FILE_PATHS, INPUT_KEYS
from src.utils import load_training_data, sample_training_data
from src.vae import VAE

SAMPLE_COUNT = 8

vae = VAE.load('model')
scaler = p.MinMaxScaler()
data = load_training_data(DATA_FILE_PATHS, INPUT_KEYS)
samples = sample_training_data(data, SAMPLE_COUNT)
x_train = scaler.fit_transform(samples)
reconstructed_data, _ = vae.reconstruct(x_train)

print('Original data:', samples)
print('Reconstructed Data:', scaler.inverse_transform(reconstructed_data))
