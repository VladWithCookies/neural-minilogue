from sklearn import preprocessing as p

from src.constants import DATA_FILE_PATHS, INPUT_KEYS
from src.utils import load_training_data, sample_data
from src.vae import VAE

SAMPLE_COUNT = 8

vae = VAE.load('model')
scaler = p.MinMaxScaler()
samples = sample_data(load_training_data(DATA_FILE_PATHS, INPUT_KEYS), SAMPLE_COUNT)
data = scaler.fit_transform(samples)
reconstructed_data, latent_representation = vae.reconstruct(data)

print('Original data: ', samples)
print('Latent Representation:', latent_representation)
print('Reconstructed Data:', scaler.inverse_transform(reconstructed_data))
