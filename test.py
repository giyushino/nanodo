#conda_env: nanodo

from nanodo.model_factory import *
from nanodo.data import *
from nanodo.configs.default import *
from nanodo.train import *
import numpy as np
from flax.linen import Partitioned
import orbax.checkpoint as ocp
from orbax.checkpoint import PyTreeCheckpointer
from flax.core import unfreeze
import jax.numpy as jnp
from nanodo.model import *


c = get_config()
#checkpoint = "/home/allanz/nanodo_workdir/92000.orbax-checkpoint-tmp-138"
checkpoint = "/home/allanz/nanodo_workdir/90000/state"
params= PyTreeCheckpointer().restore(checkpoint)
params = params['params']


test_config = ml_collections.config_dict.create(
      D=512,  # model/embed dim  = qkv dim
      F=2048,  # FF inner dimension
      H=8,  # num attention heads
      L=128,  # max context/sequence length (move out of config?)
      N=6,  # number of transformer block layers
      dtype="float32",  # cmputation dtype.
      fsdp_enabled=True,  # True to shard the model.
      remat=False,  # Transformer block gradient checkpointing to save memory.
  )
print(c.model)

tokenizer = get_py_tokenizer("tests/testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model")
vocab_size = tokenizer.GetPieceSize()
cfg = DoConfig(**test_config, V=vocab_size)  # pytype:disable=attribute-error
# model without float32
float32 = TransformerDo(cfg) 
print(float32)
# model with bfloat16
bfloat16, _ = get_model_and_loss(c, vocab_size)
print(bfloat16)
# model with jax.numpy.float32
jax_float32 = model.DoConfig(D=512, H=8, L=128, N=6, V=vocab_size, F=2048)
m = model.TransformerDo(jax_float32)
print(m)

rng = jax.random.PRNGKey(42)
_, init_rng = jax.random.split(rng)
x = jnp.ones((8, 128), dtype= jnp.int32)
initial_variables = jax.jit(bfloat16.init)(init_rng, x)


test_set = py_batched_tfds(
          tfds_name="c4_10k",
          split="train",
          context_size=128,
          worker_count=0,
          vocab_path="tests/testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model",
          batch_size = 8
          )
batch = next(iter(test_set))

def make_partitioned(array, names):
    partition_array = Partitioned(array, names = names, mesh = None)
    return partition_array

def convert_attn_blocks(params):
    blocks = ["blocks_0", "blocks_1", "blocks_2", "blocks_3", "blocks_4", "blocks_5"]
    switches = {"attn_out_proj": (None, None, 'data'), "key": ('data', None), "query":('data', None), "value":('data', None)}

    for block in blocks:
          for switch in switches: 
               #print(params[block]["CausalAttn_0"][switch]["kernel"])
               params[block]["CausalAttn_0"][switch]["kernel"] = make_partitioned(params[block]["CausalAttn_0"][switch]["kernel"]["value"], switches[switch])

def convert_Mlp(params):
    blocks = ["blocks_0", "blocks_1", "blocks_2", "blocks_3", "blocks_4", "blocks_5"]
    switches = {"Dense_0": ('data', None), "Dense_1": ('data', None)}

    for block in blocks:
          for switch in switches: 
               #print(params[block]["Mlp_0"][switch]["kernel"])
               params[block]["Mlp_0"][switch]["kernel"] = make_partitioned(params[block]["Mlp_0"][switch]["kernel"]["value"], switches[switch])


def convert_embed(params):
    params["embed"]["embedding"] = make_partitioned(params["embed"]["embedding"]["value"], (None, 'data'))

def convert_pos_embed(params):
    params["pos_embed"]["embedding"] = make_partitioned(params["pos_embed"]["embedding"]["value"], (None, 'data'))



convert_attn_blocks(params)
convert_Mlp(params)
convert_embed(params)
convert_pos_embed(params)

logits = float32.apply(initial_variables, x)
print(logits.shape)
print(logits)

# input
for x in batch:
    print(tokenizer.decode_ids(x.tolist()), "\n")

# output of the model??? 
probs = jax.nn.softmax(logits, axis=-1)
token_ids = jnp.argmax(probs, axis=-1) 

for x in token_ids:
    print(tokenizer.decode_ids(x.tolist()), "\n")