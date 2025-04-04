#conda_env: nanodo
from orbax.checkpoint import PyTreeCheckpointer
from flax.linen import Partitioned
from flax.core import unfreeze

def make_partitioned(array, names):
    """
    function to change arrays to partitioned
    """
    partition_array = Partitioned(array, names = names, mesh = None)
    return partition_array

def convert_attn_blocks(params):
    blocks = ["blocks_0", "blocks_1", "blocks_2", "blocks_3", "blocks_4", "blocks_5", "blocks_6", "blocks_7", "blocks_8", "blocks_9", "blocks_10", "blocks_11"]
    switches = {"attn_out_proj": (None, None, 'data'), "key": ('data', None, None), "query":('data', None, None), "value":('data', None, None)}

    for block in blocks:
          for switch in switches: 
               #print(params[block]["CausalAttn_0"][switch]["kernel"])
               params[block]["CausalAttn_0"][switch]["kernel"] = make_partitioned(params[block]["CausalAttn_0"][switch]["kernel"]["value"], switches[switch])

def convert_Mlp(params):
    blocks = ["blocks_0", "blocks_1", "blocks_2", "blocks_3", "blocks_4", "blocks_5", "blocks_6", "blocks_7", "blocks_8", "blocks_9", "blocks_10", "blocks_11"]
    switches = {"Dense_0": ('data', None), "Dense_1": ('data', None)}

    for block in blocks:
          for switch in switches: 
               #print(params[block]["Mlp_0"][switch]["kernel"])
               params[block]["Mlp_0"][switch]["kernel"] = make_partitioned(params[block]["Mlp_0"][switch]["kernel"]["value"], switches[switch])

def convert_embed(params):
    params["embed"]["embedding"] = make_partitioned(params["embed"]["embedding"]["value"], (None, 'data'))

def convert_pos_embed(params):
    params["pos_embed"]["embedding"] = make_partitioned(params["pos_embed"]["embedding"]["value"], (None, 'data'))

def reformat_params(params):
    params = unfreeze(params)
    convert_attn_blocks(params)
    convert_Mlp(params)
    convert_embed(params)
    convert_pos_embed(params)
    return params

def load_params(checkpoint):
    params = PyTreeCheckpointer().restore(checkpoint)['params']
    new_params = reformat_params(params)
    return new_params

if __name__ == "__main__":
    checkpoint = "/home/allanz/nanodo_workdir/44000/state"
    test = load_params(checkpoint)
    print(test)
