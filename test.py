from orbax.checkpoint import PyTreeCheckpointer
from nanodo.configs import default
from nanodo import model_factory
from nanodo import data_custom
from nanodo import params as par
from nanodo import model
from nanodo import data
import jax.numpy as jnp
import ml_collections
from jax import nn
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

c = default.get_config()
tokenizer = data.get_py_tokenizer(c.vocab_path)
vocab_size = tokenizer.GetPieceSize()
model, _ = model_factory.get_model_and_loss(c, vocab_size)
print(model)

checkpoint = "/home/allanz/nanodo_workdir/44000/state"
params = par.load_params(checkpoint)

dataset = data.py_batched_tfds(
          tfds_name="c4",
          split="train",
          context_size=1024,
          worker_count=0,
          vocab_path=c.vocab_path,
          batch_size=32,
      )

data = next(iter(dataset))
logits = model.apply({'params':params}, data)

probabilities = nn.softmax(logits, axis=-1)
decoded_indices = jnp.argmax(probabilities, axis=-1)
decoded_texts = [tokenizer.decode_ids(sequence.tolist()) for sequence in decoded_indices]

# decoded_texts will be a list of 32 decoded strings
for i, text in enumerate(decoded_texts):
    print(f"Decoded Text {i + 1}: {text}")




last_token_indices = decoded_indices[:, -1] 
decoded_texts = [tokenizer.decode_ids(sequence.tolist()) for sequence in last_token_indices]

# decoded_texts will be a list of 32 decoded strings
for i, text in enumerate(decoded_texts):
    print(f"Decoded Text {i + 1}: {text}")





