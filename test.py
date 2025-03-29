#conda_env: nanodo

from nanodo.configs import default
from nanodo import model_factory
from nanodo import data_custom
from nanodo import model
from nanodo import data
import jax.numpy as jnp
import time
import jax

# get configs used during training

if __name__ == "__main__":
    initial = time.time()
    c = default.get_config()
    print(c)

    # load tokenizer
    tokenizer = data.get_py_tokenizer(c.vocab_path)
    vocab_size = tokenizer.GetPieceSize()
    print(f"vocab size: {vocab_size}")

    model, _ = model_factory.get_model_and_loss(c, vocab_size)

    # load in dataset
    print("loading dataset")

    t0 = time.time()
    train_set = data.py_batched_tfds(
              tfds_name="c4",
              split="train",
              context_size=1024,
              worker_count=16, # number of pygrain workers? 
              vocab_path="tests/testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model",
              batch_size = 8
              )
    t1 = time.time()
    print(f"{(t1 - t0):.6f} seconds to load c4")
    #batch = next(iter(train_set))
    train_batch = next(iter(train_set))
    train_batch_2 = next(iter(train_set))
    # load the model params
    t2 = time.time()
    rng = jax.random.PRNGKey(42)
    _, init_rng = jax.random.split(rng)
    x = jnp.ones((8, 1024), dtype= jnp.int32)
    initial_variables = jax.jit(model.init)(init_rng, x)
    t3 = time.time()
    print(f"{(t3 - t2):.6f} seconds to create initial variables")

    t3 = time.time()
    test_logits = model.apply(initial_variables, train_batch)
    test_logits_2 = model.apply(initial_variables, train_batch_2)
    t4 = time.time()
    print(test_logits)
    print(test_logits.shape)
    print(f"{(t4 - t3):.6f} seconds to run inference")
    final = time.time()
    print(final - initial)



