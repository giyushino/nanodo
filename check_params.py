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
c = default.get_config()
bfloat16, loss_fn = model_factory.get_model_and_loss(c, vocab_size)
print(bfloat16)
print("loaded model")

checkpoint = "/home/allanz/nanodo_workdir/44000/state"

ones = jnp.ones((32, 1024), dtype=jnp.int32)
#print(ones)
test_params = par.load_params(checkpoint)
with open("checkpoint_formatted.txt", "w") as file:
    file.write(f"{test_params}")
test = bfloat16.apply({'params': test_params}, ones)
print(test)






