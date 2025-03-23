from datasets import load_dataset
import time


def load_c4():
    dataset = load_dataset("allenai/c4", "en")
    dataset = dataset.remove_columns(["url"])
    dataset = dataset.remove_columns(["timestamp"])
    return dataset

def load_c4_10k():
    return load_dataset("stas/c4-en-10k")

def encode_text(dataset):
    return {'text':dataset["text"].encode("utf-8")}

def load_custom(dataset):
    if dataset == "c4":
        return load_c4()
    elif dataset == "c4_10k":
        return load_c4_10k()
    else: 
        print("Invalid dataset name")


if __name__ == "__main__":
    t0 = time.perf_counter()
    hf_dataset = load_c4()
    t1 = time.perf_counter()
    hf_dataset_10k = load_c4_10k()
    t2 = time.perf_counter()
    print(f"{(t1 - t0):.6f} to load c4")
    print(f"{(t2 - t1):.6f} to load c4_10k")
    print(hf_dataset["train"][0])
    print(hf_dataset_10k["train"][0])



