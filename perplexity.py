#conda_env: nanodo

from nanodo.model_factory import *
from nanodo.data import *
from nanodo.configs.default import *
import ml_collections
from nanodo import model 
from nanodo.params import *
import numpy as np
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import PyTreeCheckpointer


c = get_config()
tokenizer = get_py_tokenizer("tests/testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model")
vocab_size = tokenizer.GetPieceSize()
transformer_decoder, get_loss_fn = get_model_and_loss(c, vocab_size)
print(transformer_decoder)

checkpoint_folder = "/tmp/nanodo_workdir/68000/state"
params = PyTreeCheckpointer().restore(checkpoint_folder)
print(params)

tokens = tokenizer.encode('i love computer science and', out_type=int, enable_sampling=True, alpha = 0.1, nbest_size = -1)
input_array = jnp.array([tokens])
print(input_array.shape)
logits = transformer_decoder.apply(params, input_array)
print(logits.shape)
predicted_tokens = jnp.argmax(logits, axis=-1)
output_text = tokenizer.decode(predicted_tokens[0].tolist())
print(output_text)



"""
model = ml_collections.config_dict.create(
  D=512,  # model/embed dim  = qkv dim
  H=8,  # num attention heads
  L=512,  # max context/sequence length (move out of config?)
  N=6,  # number of transformer block layers
  F=2048,  # FF inner dimension
  dtype="bfloat16",  # computation dtype.
  fsdp_enabled=True,  # True to shard the model.
  remat=False,  # Transformer block gradient checkpointing to save memory.
)


print(tokenizer)
print(vocab_size)

test, test_loss = get_model_and_loss(model, vocab_size)
"""
