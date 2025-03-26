#conda_env: nanodo

from orbax.checkpoint import PyTreeCheckpointer
from nanodo.configs import default
from nanodo import model_factory
from nanodo import data_custom
from nanodo import params as par
from nanodo import model
from nanodo import data
import jax.numpy as jnp
import ml_collections

# load tokenizer
vocab_path ="tests/testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model"
sp_tok= data.get_py_tokenizer(vocab_path)
nanodo_tok = data._SPTokenizer(vocab_path)
vocab_size = sp_tok.GetPieceSize() # got piece size from tokenizer 
print(vocab_size)
 
# load config, this is the same as used in training
test_config = ml_collections.config_dict.create(
      D=512,  # model/embed dim  = qkv dim
      F=2048,  # FF inner dimension
      H=8,  # num attention heads
      L=128,  # max context/sequence length (move out of config?)
      N=6,  # number of transformer block layers
      V=32101,
      dtype="float32",  # computation dtypesh
      fsdp_enabled=True,  # True to shard the model.
      remat=False,  # Transformer block gradient checkpointing to save memory.
  )
print(test_config)
# model loaded manually
cfg = model.DoConfig(**test_config)
float32 = model.TransformerDo(cfg)
print(float32)

# model loaded through model factory
c = default.get_config()
bfloat16, loss_fn = model_factory.get_model_and_loss(c, 32101)
print(bfloat16)

checkpoint = "/home/allanz/nanodo_workdir/58000/state"
params = PyTreeCheckpointer().restore(checkpoint)
params = params['params']
#print(params)

ones = jnp.ones((8, 128), dtype=jnp.int32)
#print(ones)

test_params = par.load_params(checkpoint)
#print(test_params)

test = float32.apply({'params': test_params}, ones)
print(test)




