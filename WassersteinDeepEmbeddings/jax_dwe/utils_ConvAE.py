import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state 
from clu import metrics

import jax
import jax.numpy as jnp
from jax import random


from functools import partial
import scipy.stats
import numpy as np

from typing import Callable, Any, Optional

@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    vocab_size: int
    output_vocab_size: int
    dtype: Any = jnp.float32
    dist_func_enc: str = 'S2'
    dist_func_dec: str = 'S2'
    eps_enc: float = 0.1
    eps_dec: float = 0.01
    lse_enc: bool = False
    lse_dec: bool = True
    coeff_dec: float = 1
    scale: str = 'min_max_total'
    factor: float = 1.0
    emb_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    qkv_dim: int = 128
    mlp_dim: int = 512
    max_len: int = 256
    attention_dropout_rate: float = 0.1
    kernel_init: Callable = nn.initializers.glorot_uniform()
    bias_init: Callable = nn.initializers.zeros_init()



class ConvAE_3D(nn.Module):

    
   
    
    enc_dim: int
    inp_shape: int
    
    @nn.compact
    def __call__(self, x):
        
        x = x[:, :, :, :, None]
        
        enc_dim = self.enc_dim
        inp_shape = self.inp_shape
        
        batch_size = x.shape[0]
        # Encoder:
        # Only 2 convolution and 2 max pooling layers are used in alternating order.

        x = nn.Conv(features=32, kernel_size=(3, 3, 3), strides=1, padding="SAME")(x)  # 28x28x1 --> 28x28x32
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3, 3), strides=1, padding="SAME")(x)  # 28x28x1 --> 28x28x32
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2, 2), strides=(2, 2, 2))  # 28x28x32 --> 14x14x32

        x = nn.Conv(features=64, kernel_size=(3, 3, 3), strides=1, padding="SAME")(x)  # 14x14x32 --> 14x14x64
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3, 3), strides=1, padding="SAME")(x)  # 14x14x32 --> 14x14x64
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2, 2), strides=(2, 2, 2))  # 28x28x32 --> 14x14x32

        x = x.reshape([batch_size, -1])
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        x = nn.Dense(256)(x)
        x = nn.relu(x)

        enc = nn.Dense(enc_dim)(x)

        x = nn.Dense(256)(enc)
        x = nn.relu(x)
        
        x = nn.Dense(int(inp_shape/4)**3)(x)
        x = nn.relu(x)
        
        x = x.reshape([batch_size, int(inp_shape/4), int(inp_shape/4),  int(inp_shape/4), 1])

        x = nn.ConvTranspose(features=32, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="SAME")(x) # 7x7x64 --> 14x14x32
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3 ,3), strides=1, padding="SAME")(x)  # 14x14x32 --> 14x14x32
        x = nn.relu(x)

        x = nn.ConvTranspose(features=16, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="SAME")(x) # 14x14x32 --> 28x28x16
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3, 3), strides=1, padding="SAME")(x)  # 28x28x16 --> 28x28x16
        x = nn.relu(x)
        
        dec = nn.Conv(features=1, kernel_size=(3, 3, 3), strides=1, padding="SAME")(x)  # 28x28x16 --> 28x28x1
        dec = jnp.squeeze(dec, axis = -1)
        
        return enc, dec
    
class ConvAE_2D(nn.Module):

    

    
    enc_dim: int
    inp_shape: int
    
    @nn.compact
    def __call__(self, x):
        
        x = x[:, :, :, None]
        
        enc_dim = self.enc_dim
        inp_shape = self.inp_shape
        # Encoder:
        # Only 2 convolution and 2 max pooling layers are used in alternating order.

        batch_size = x.shape[0]
        
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding="SAME")(x)  
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding="SAME")(x)  
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2, 2)) 

        x = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding="SAME")(x)  
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding="SAME")(x)  
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2, 2))  
        
        x = x.reshape([batch_size, -1])
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        x = nn.Dense(256)(x)
        x = nn.relu(x)

        enc = nn.Dense(enc_dim)(x)

        x = nn.Dense(256)(enc)
        x = nn.relu(x)
        
        x = nn.Dense(int(inp_shape/4)**2)(x)
        x = nn.relu(x)
        
        x = x.reshape([batch_size, int(inp_shape/4), int(inp_shape/4), 1])

        x = nn.ConvTranspose(features=32, kernel_size=(2, 2), strides=(2, 2), padding="SAME")(x) 
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding="SAME")(x)  
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding="SAME")(x)  
        x = nn.relu(x)

        x = nn.ConvTranspose(features=16, kernel_size=(2, 2), strides=(2, 2), padding="SAME")(x) 
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=1, padding="SAME")(x)  
        x = nn.relu(x)
        dec = nn.Conv(features=1, kernel_size=(3, 3), strides=1, padding="SAME")(x)  
        dec = jnp.squeeze(dec, axis = -1)
        
        return enc, dec

    
@struct.dataclass
class Metrics(metrics.Collection):
    enc_loss: metrics.Average.from_output('enc_loss')
    dec_loss: metrics.Average.from_output('dec_loss')
    enc_corr: metrics.Average.from_output('enc_corr')
    
class TrainState(train_state.TrainState):
    metrics: Metrics

    

        