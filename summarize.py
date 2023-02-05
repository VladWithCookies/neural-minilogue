from src.vae import VAE

vae = VAE(
  input_shape = (39,),
  hidden_nodes = (16,),
  latent_nodes = 8
)

vae.summary()
