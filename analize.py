from sklearn import preprocessing as p

from src.constants import DATA_FILE_PATHS, INPUT_KEYS
from src.utils import load_data, sample_data
from src.vae import VAE

SAMPLE_COUNT = 8

vae = VAE.load('model')
scaler = p.MinMaxScaler()
data = load_data(DATA_FILE_PATHS, INPUT_KEYS)
samples = sample_data(data, SAMPLE_COUNT)
x_train = scaler.fit_transform(samples)
reconstructed_data, latent_representation = vae.reconstruct(x_train)
denormalized_data = scaler.inverse_transform(reconstructed_data)

print('Original data:', samples)
print('Latent representation:', latent_representation)
print('Reconstructed data:', denormalized_data)
