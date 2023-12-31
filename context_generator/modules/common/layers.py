import jax
import jax.numpy as jnp
import haiku as hk

import dataclasses
from typing import Optional

@dataclasses.dataclass
class AngularEncoding(hk.Module):

    num_funcs: int = 3
    name: Optional[str] = None

    def get_out_dim(self, in_dim: int):
        return in_dim * (1 + 2*2*self.num_funcs)

    def __call__(
            self,
            x: jnp.DeviceArray    # x: (..., d)
    ) -> jnp.DeviceArray:
        
        freq_bands = jnp.array(
            [i+1 for i in range(self.num_funcs)] + [1./(i+1) for i in range(self.num_funcs)]
        )
        shape = list(x.shape[:-1]) + [-1]
        x = jnp.expand_dims(x, -1)
        code = jnp.concatenate([x, jnp.sin(x * freq_bands), jnp.cos(x * freq_bands)], axis=-1)    # (..., d, 2f+1)
        code = code.reshape(shape)
        return code
    
    

def mask_zero(mask, value):
    return jnp.where(mask, value, jnp.zeros_like(value))