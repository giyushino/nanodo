#conda_env: nanodo
import tensorflow_datasets as tfds
from nanodo.c4_dataset import *
from nanodo.data import *
import nanodo.data_custom as dc
import time
t0 = time.perf_counter()
tf_dataset = tfds.data_source("lm1b:1.1.0", split = "train")


print("lm1b dataset")
print(tf_dataset)
print(next(iter(tf_dataset)))
print("\n")

ds = py_batched_tfds(
  tfds_name="lm1b",
  split="train",
  context_size=1024,
  worker_count=0,
  vocab_path="tests/testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model",
  batch_size=8,
)

print(next(iter(ds)))
print("==================")




ds = dc.py_batched_tfds(
  tfds_name="lm1b",
  split="train",
  context_size=1024,
  worker_count=0,
  vocab_path="tests/testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model",
  batch_size=8,
)
print(next(iter(ds)))
t1 = time.perf_counter()
print(t1 - t0)


