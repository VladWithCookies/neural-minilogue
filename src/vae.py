import os
import pickle
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.legacy import Adam

tf.compat.v1.disable_eager_execution()

class VAE:
  def __init__(self, input_shape, hidden_nodes, latent_nodes):
    self.input_shape = input_shape
    self.hidden_nodes = hidden_nodes
    self.latent_nodes = latent_nodes
    self.reconstruction_loss_weight = 1000

    self.encoder = None
    self.decoder = None
    self.model = None

    self._shape_before_bottleneck = None
    self._model_input = None

    self._build()

  @classmethod
  def load(cls, folder = '.'):
    params_path = os.path.join(folder, 'params.pkl')

    with open(params_path, 'rb') as file:
      params = pickle.load(file)

    vae = VAE(*params)
    weights_path = os.path.join(folder, 'weights.h5')
    vae.load_weights(weights_path)

    return vae

  def _calculate_combined_loss(self, y_target, y_predicted):
    reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
    kl_loss = self._calculate_kl_loss(y_target, y_predicted)

    return self.reconstruction_loss_weight * reconstruction_loss + kl_loss

  def _calculate_reconstruction_loss(self, y_target, y_predicted):
    error = y_target - y_predicted

    return K.mean(K.square(error), axis = 1)

  def _calculate_kl_loss(self, y_target, y_predicted):
    return -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) - K.exp(self.log_variance), axis = 1)

  def _create_folder(self, folder):
    if not os.path.exists(folder):
      os.makedirs(folder)

  def _save_parameters(self, folder):
    params = [
      self.input_shape,
      self.hidden_nodes,
      self.latent_nodes
    ]

    path = os.path.join(folder, 'params.pkl')

    with open(path, 'wb') as file:
      pickle.dump(params, file)

  def _save_weights(self, folder):
    path = os.path.join(folder, 'weights.h5')

    self.model.save_weights(path)

  def _build(self):
    self._build_encoder()
    self._build_decoder()
    self._build_vae()

  def _build_vae(self):
    model_input = self._model_input
    model_output = self.decoder(self.encoder(model_input))

    self.model = Model(model_input, model_output, name = 'vae')

  def _build_encoder(self):
    encoder_input = self._add_encoder_input()
    hidden_layers = self._add_encoder_hidden_layers(encoder_input)
    bottleneck = self._add_bottleneck(hidden_layers)

    self._model_input = encoder_input
    self.encoder = Model(encoder_input, bottleneck, name = 'encoder')

  def _add_encoder_input(self):
    return Input(shape = self.input_shape, name = 'encoder_input')

  def _add_encoder_hidden_layers(self, encoder_input):
    graph = encoder_input

    for index in range(len(self.hidden_nodes)):
      graph = Dense(self.hidden_nodes[index], name = f"encoder_hidden_layer_{index}")(graph)

    return graph

  def _add_bottleneck(self, graph):
    self._shape_before_bottleneck = K.int_shape(graph)[1:]
    self.mu = Dense(self.latent_nodes, name = 'mu')(graph)
    self.log_variance = Dense(self.latent_nodes, name = 'log_variance')(graph)

    def sample_point_from_normal_distribution(args):
      mu, log_variance = args
      epsilon = K.random_normal(shape = K.shape(self.mu), mean = 0., stddev = 1.)

      return mu + K.exp(log_variance / 2) * epsilon

    return Lambda(
      sample_point_from_normal_distribution,
      name = 'encoder_output'
    )([self.mu, self.log_variance])

  def _build_decoder(self):
    decoder_input = self._add_decoder_input()
    hidden_layers = self._add_decoder_hidden_layers(decoder_input)
    decoder_output = self._add_decoder_output(hidden_layers)

    self.decoder = Model(decoder_input, decoder_output, name = 'decoder')

  def _add_decoder_input(self):
    return Input(shape = self.latent_nodes, name = 'decoder_input')

  def _add_decoder_hidden_layers(self, decoder_input):
    graph = decoder_input

    for index in reversed(range(len(self.hidden_nodes))):
      graph = Dense(self.hidden_nodes[index], name = f"decoder_hidden_layer_{index}")(graph)

    return graph

  def _add_decoder_output(self, graph):
    return Dense(self.input_shape[0], name = 'decoder-output')(graph)

  def compile(self, learning_rate = 0.0001):
    optimizer = Adam(learning_rate = learning_rate)

    def calculate_reconstruction_loss(y_target, y_predicted):
      return self._calculate_combined_loss(y_target, y_predicted)

    def calculate_kl_loss(y_target, y_predicted):
      return self._calculate_kl_loss(y_target, y_predicted)

    self.model.compile(
      optimizer = optimizer,
      loss = self._calculate_combined_loss,
      metrics = [calculate_reconstruction_loss, calculate_kl_loss]
    )

  def train(self, x_train, batch_size, epochs):
    self.model.fit(x_train, x_train, batch_size = batch_size, epochs = epochs, shuffle = True)

  def save(self, folder = '.'):
    self._create_folder(folder)
    self._save_parameters(folder)
    self._save_weights(folder)

  def load_weights(self, path):
    self.model.load_weights(path)

  def reconstruct(self, data):
    latent_representations = self.encoder.predict(data)
    reconstructed_data = self.decoder.predict(latent_representations)

    return reconstructed_data, latent_representations

  def summary(self):
    self.encoder.summary()
    self.decoder.summary()
    self.model.summary()
