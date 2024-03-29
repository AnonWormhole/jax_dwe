
import optax
from flax import linen as nn
from flax import struct
from flax.training import train_state 

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from functools import partial
import scipy.stats
import numpy as np
from tqdm import trange

from jax_dwe.utils_ConvAE import * 
from jax_dwe.utils_OT import * 



def MaxMinScale(arr):
    min_arr = arr.min(axis = 0)
    max_arr = arr.max(axis = 0)
    
    arr = 2*(arr - arr.min(axis = 0, keepdims = True))/(arr.max(axis = 0, keepdims = True) - arr.min(axis = 0, keepdims = True))-1
    return(arr)

def pad_pointclouds(point_clouds, max_shape = -1):
    if(max_shape == -1):
        max_shape = np.max([pc.shape[0] for pc in point_clouds])+1
    else:
        max_shape = max_shape + 1
    weight_vec = np.asarray([np.concatenate((np.ones(pc.shape[0]), np.zeros(max_shape - pc.shape[0])), axis = 0) for pc in point_clouds])
    point_clouds_pad = np.asarray([np.concatenate([pc, np.zeros([max_shape - pc.shape[0], pc.shape[-1]])], axis = 0) for pc in point_clouds])
    

    return(point_clouds_pad[:, :-1].astype('float32'), weight_vec[:, :-1].astype('float32'))

def voxelize(point_clouds, num_vox_per_axis, min_val = None, max_val = None):
    if(min_val is None):
        min_val = np.min([pc.min() for pc in point_clouds])
        max_val = np.max([pc.max() for pc in point_clouds])
        
    pc_norm_cord = [((num_vox_per_axis-1) * np.clip((pc-min_val)/(max_val - min_val), 0, 1)).astype('int') for pc in point_clouds]
    
    dim = point_clouds[0].shape[1]
    if(dim == 2):
        images = np.zeros([len(point_clouds), num_vox_per_axis, num_vox_per_axis])
        for ind, pc in enumerate(pc_norm_cord):
            images[ind][pc[:, 0], pc[:, 1]] += 1
            images[ind] = images[ind]/images[ind].sum()
    else:
        images = np.zeros([len(point_clouds), num_vox_per_axis, num_vox_per_axis, num_vox_per_axis])
        for ind, pc in enumerate(pc_norm_cord):
            images[ind][pc[:, 0], pc[:, 1], pc[:, 2]] += 1
            images[ind] = images[ind]/images[ind].sum()
    return(images, min_val, max_val)

class jax_dwe():

    def __init__(self, point_clouds, point_clouds_test = None, vox_per_axis = 28, key = random.key(0), config = Config):
    


        self.config = config
        self.point_clouds = point_clouds
        self.dim = self.point_clouds[0].shape[-1]
        vox_per_axis = int(np.ceil(vox_per_axis/4))*4
        
        if(point_clouds_test is None):
            self.images, _, _ = voxelize(self.point_clouds, vox_per_axis)
                
            self.point_clouds, self.masks = pad_pointclouds(self.point_clouds)
            self.masks_normalized = self.masks/self.masks.sum(axis = 1, keepdims = True)
            
           
        else:
            self.point_clouds_test = point_clouds_test
            
            self.images, min_val, max_val = voxelize(self.point_clouds, vox_per_axis)
            self.test_images, _, _ = voxelize(self.point_clouds_test, vox_per_axis, min_val, max_val) 
                
                
            total_point_clouds, total_masks = pad_pointclouds(list(self.point_clouds) + list(self.point_clouds_test))
            self.point_clouds, self.masks = total_point_clouds[:len(list(self.point_clouds))], total_masks[:len(list(self.point_clouds))]
            self.point_clouds_test, self.masks_test = total_point_clouds[len(list(self.point_clouds)):], total_masks[len(list(self.point_clouds)):]

            self.masks_normalized = self.masks/self.masks.sum(axis = 1, keepdims = True)
            self.masks_test_normalized = self.masks_test/self.masks_test.sum(axis = 1, keepdims = True)

        self.out_seq_len = int(jnp.median(jnp.sum(self.masks, axis = 1)))
        self.inp_dim = self.point_clouds.shape[-1]

        

        
        self.eps_enc = config.eps_enc
        self.eps_dec = config.eps_dec

        self.lse_enc = config.lse_enc
        self.lse_dec = config.lse_dec

        self.coeff_dec = config.coeff_dec
        
        self.dist_func_enc = config.dist_func_enc
        if(self.dist_func_enc == 'W1'):
            self.jit_dist_enc = jax.jit(jax.vmap(W1, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_enc == 'W2'):
            self.jit_dist_enc = jax.jit(jax.vmap(W2, (0, 0, None, None), 0), static_argnums=[2,3])    
        if(self.dist_func_enc == 'W2_norm'):
            self.jit_dist_enc = jax.jit(jax.vmap(W2_norm, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_enc == 'GW'):
            self.jit_dist_enc = jax.jit(jax.vmap(GW, (0, 0, None, None), 0), static_argnums=[2,3]) 
        if(self.dist_func_enc == 'S1'):
            self.jit_dist_enc = jax.jit(jax.vmap(S1, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_enc == 'S2'):
            self.jit_dist_enc = jax.jit(jax.vmap(S2, (0, 0, None, None), 0), static_argnums=[2,3])    
        if(self.dist_func_enc == 'S2_norm'):
            self.jit_dist_enc = jax.jit(jax.vmap(S2_norm, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_enc == 'GS'):
            self.jit_dist_enc = jax.jit(jax.vmap(GS, (0, 0, None, None), 0), static_argnums=[2,3]) 

        self.dist_func_dec = config.dist_func_dec
        if(self.dist_func_dec == 'W1'):
            self.jit_dist_dec = jax.jit(jax.vmap(W1_grad, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_dec == 'W2'):
            self.jit_dist_dec = jax.jit(jax.vmap(W2_grad, (0, 0, None, None), 0), static_argnums=[2,3])    
        if(self.dist_func_dec == 'W2_norm'):
            self.jit_dist_dec = jax.jit(jax.vmap(W2_norm_grad, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_dec == 'GW'):
            self.jit_dist_dec = jax.jit(jax.vmap(GW_grad, (0, 0, None, None), 0), static_argnums=[2,3]) 
        if(self.dist_func_dec == 'S1'):
            self.jit_dist_dec = jax.jit(jax.vmap(S1, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_dec == 'S2'):
            self.jit_dist_dec = jax.jit(jax.vmap(S2, (0, 0, None, None), 0), static_argnums=[2,3])    
        if(self.dist_func_dec == 'S2_norm'):
            self.jit_dist_dec = jax.jit(jax.vmap(S2_norm, (0, 0, None, None), 0), static_argnums=[2,3])
        if(self.dist_func_dec == 'GS'):
            self.jit_dist_dec = jax.jit(jax.vmap(GS, (0, 0, None, None), 0), static_argnums=[2,3]) 
        if(self.coeff_dec < 0):
            self.jit_dist_dec  = jax.jit(jax.vmap(Zeros, (0, 0, None, None), 0), static_argnums=[2,3]) 
            self.coeff_dec = 0.0
      


        self.scale = config.scale
        self.factor = config.factor
        self.point_clouds = self.scale_func(self.point_clouds) * self.factor
        if(point_clouds_test is not None):
            self.point_clouds_test = self.scale_func(self.point_clouds_test)*self.factor
        
        if(self.dim == 2):
            self.model = ConvAE_2D(enc_dim = config.emb_dim, inp_shape = self.images.shape[-1])
        else:
            self.model = ConvAE_3D(enc_dim = config.emb_dim, inp_shape = self.images.shape[-1])

    def scale_func(self, point_clouds):
        if(self.scale == 'max_dist_total'):
            if(not hasattr(self, 'max_scale_num')):
                max_dist = 0
                for _ in range(10):
                    i,j = np.random.choice(np.arange(len(self.point_clouds)), 2,replace = False)
                    if(self.dist_func_enc == 'GW' or self.dist_func_enc == 'GS'):
                        max_ij = np.max(scipy.spatial.distance.cdist(self.point_clouds[i], self.point_clouds[i])**2)
                    else:
                        max_ij = np.max(scipy.spatial.distance.cdist(self.point_clouds[i], self.point_clouds[j])**2)
                    max_dist = np.maximum(max_ij, max_dist)
                self.max_scale_num = max_dist
            else:
                print("Using Calculated Max Dist Scaling Values") 
            return(point_clouds/self.max_scale_num)
        if(self.scale == 'max_dist_each'):
            print("Using Per Sample Max Dist") 
            pc_scale = np.asarray([pc/np.max(scipy.spatial.distance.pdist(pc)**2) for pc in point_clouds])
            return(pc_scale)
        if(self.scale == 'min_max_each'):
            print("Scaling Per Sample") 
            max_val = point_clouds.max(axis = 1, keepdims = True)
            min_val = point_clouds.min(axis = 1, keepdims = True)
            return(2 * (point_clouds - min_val)/(max_val - min_val) - 1)
        elif(self.scale == 'min_max_total'):
            if(not hasattr(self, 'max_val')):
                self.max_val = self.point_clouds.max(axis = ((0,1)), keepdims = True)
                self.min_val = self.point_clouds.min(axis = ((0,1)), keepdims = True)
            else:
                print("Using Calculated Min Max Scaling Values") 
            return(2 * (point_clouds - self.min_val)/(self.max_val - self.min_val) - 1)
        elif(isinstance(self.scale, (int, float, complex)) and not isinstance(self.scale, bool)):
            print("Using Constant Scaling Value") 
            return(point_clouds/self.scale)
    
    def encode(self, images, max_batch = 256):
        if(images.shape[0] < max_batch):
            enc, _ = self.model.apply(variables = {'params': self.params}, x = images)
        else: # For when the GPU can't pass all point-clouds at once
            num_split = int(images.shape[0]/max_batch)+1
            images_split = np.array_split(images, num_split)
            enc = np.concatenate([self.model.apply({'params': self.params}, x = images_split[split_ind])[0] for
                                  split_ind in range(num_split)], axis = 0)
        return enc
    
    def decode(self, images, max_batch = 256):
        if(images.shape[0] < max_batch):
            _, dec = self.model.apply(variables = {'params':self.params}, x = images)
        else: # For when the GPU can't pass all point-clouds at once
            num_split = int(images.shape[0]/max_batch)+1
            images_split = np.array_split(images, num_split)
            dec = np.concatenate([self.model.apply({'params': self.params}, x = images_split[split_ind])[1] for
                                  split_ind in range(num_split)], axis = 0)
        return dec
    
    
    def kl_div(self, logits, labels):
        y_pred = jax.nn.softmax(logits.reshape([logits.shape[0], -1]))
        y_true = labels.reshape([labels.shape[0], -1])
    
        y_true = jnp.clip(y_true, 0.00001, 1)
        y_pred = jnp.clip(y_pred, 0.00001, 1)
        
        return(jnp.mean(y_true * jnp.log(y_true / y_pred), axis = -1))


    def compute_losses(self, pc_x, pc_y, 
                       masks_x, masks_y, 
                       images_x, images_y, 
                       enc_x, enc_y, 
                       dec_x, dec_y):
        
        masks_x_normalized = masks_x/jnp.sum(masks_x, axis = 1,keepdims = True)
        masks_y_normalized = masks_y/jnp.sum(masks_y, axis = 1,keepdims = True)

        pc_pairwise_dist = self.jit_dist_enc([pc_x, masks_x_normalized],
                                             [pc_y, masks_y_normalized], 
                                             self.eps_enc, self.lse_enc)
        enc_pairwise_dist = jnp.mean(jnp.square(enc_x - enc_y), axis = 1)
        
        dec_dist_x = self.kl_div(dec_x, images_x)
        dec_dist_y = self.kl_div(dec_y, images_y)
        
        return(pc_pairwise_dist, enc_pairwise_dist, dec_dist_x, dec_dist_y)
       
    
    def create_train_state(self, key = random.key(0), init_lr = 0.0001, decay_steps = 2000):
        
        key, subkey = random.split(key)
        params = self.model.init(rngs = {'params': key},
                                         x = self.images[0:5])['params']
        
        lr_sched = optax.exponential_decay(0.0001, decay_steps, 0.9, staircase = True)
        tx = optax.adam(lr_sched)#
        
        return(TrainState.create(
          apply_fn=self.model.apply, params=params, tx=tx,
          metrics=Metrics.empty()))
    
    @partial(jit, static_argnums=(0, ))
    def train_step(self, state, 
                   pc_x, pc_y, 
                   masks_x, masks_y,
                   images_x, images_y):
        """Train for a single step."""
        
        def loss_fn(params):
            enc_x, dec_x = state.apply_fn({'params':params}, x = images_x)
            enc_y, dec_y = state.apply_fn({'params':params}, x = images_y)
            
            pc_pairwise_dist, enc_pairwise_dist, dec_dist_x, dec_dist_y = self.compute_losses(pc_x, pc_y, 
                                                                                              masks_x, masks_y, 
                                                                                              images_x, images_y, 
                                                                                              enc_x, enc_y, 
                                                                                              dec_x, dec_y)
            
            enc_loss = jnp.mean(jnp.square(pc_pairwise_dist - enc_pairwise_dist))
            dec_loss = 0.5 * jnp.mean(dec_dist_x) +  0.5 * jnp.mean(dec_dist_y)
            enc_corr = jnp.corrcoef(enc_pairwise_dist, pc_pairwise_dist)[0,1]
            return(enc_loss + self.coeff_dec * dec_loss, [enc_loss, dec_loss, enc_corr])
    
        grad_fn = jax.value_and_grad(loss_fn, has_aux = True)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return(state, loss)


    def train(self, epochs = 10000, batch_size = 16, verbose = 8, init_lr = 0.0001, decay_steps = 2000, key = random.key(0)):
        batch_size = min(self.point_clouds.shape[0], batch_size)
        batch_size = int(batch_size/2)*2
        
        self.tri_u_ind = jnp.stack(jnp.triu_indices(batch_size, 1), axis =1)
        self.pseudo_masks = jnp.ones([batch_size, self.out_seq_len])/self.out_seq_len

        key, subkey = random.split(key)
        state = self.create_train_state(subkey, init_lr = init_lr, decay_steps = decay_steps)
        
        self.params = state.params
        
        self.enc_loss, self.dec_loss = [], []
        
        tq = trange(epochs, leave=True, desc = "")
        enc_loss_mean, dec_loss_mean, enc_corr_mean, count = 0,0,0,0
        for epoch in tq:
            # time.sleep(1)
            key, subkey = random.split(key)
            
            batch_ind = random.choice(key = subkey, a = self.point_clouds.shape[0], shape = [batch_size], replace = False)
            point_clouds_batch, masks_batch, images_batch = self.point_clouds[batch_ind], self.masks[batch_ind], self.images[batch_ind]
    
            pc_x, pc_y = point_clouds_batch[:int(batch_size/2)], point_clouds_batch[int(batch_size/2):]
            masks_x, masks_y = masks_batch[:int(batch_size/2)], masks_batch[int(batch_size/2):]
            images_x, images_y = images_batch[:int(batch_size/2)], images_batch[int(batch_size/2):]
            
            key, subkey = random.split(key)
            state, loss = self.train_step(state, pc_x, pc_y, 
                                                 masks_x, masks_y,
                                                 images_x, images_y)
            self.params = state.params

            enc_loss_mean, dec_loss_mean, enc_corr_mean, count = enc_loss_mean + loss[1][0], dec_loss_mean + loss[1][1], enc_corr_mean + loss[1][2], count + 1
            
            self.enc_loss.append(loss[1][0])
            self.dec_loss.append(loss[1][1])
            
            if(epoch%verbose==0):
                print_statement = ''
                for metric,value in zip(['enc_loss', 'dec_loss', 'enc_corr'], [enc_loss_mean, dec_loss_mean, enc_corr_mean]):
                    if(metric == 'enc_corr'):
                        print_statement = print_statement + ' ' + metric + ': {:.3f}'.format(value/count)
                    else:
                        print_statement = print_statement + ' ' + metric + ': {:.3e}'.format(value/count)

                # state.replace(metrics=state.metrics.empty())
                enc_loss_mean, dec_loss_mean, enc_corr_mean, count = 0,0,0,0
                tq.set_description(print_statement)
                tq.refresh() # to show immediately the update
            
