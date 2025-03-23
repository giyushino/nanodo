#conda_env: nanodo
import orbax.checkpoint as ocp
from orbax.checkpoint import PyTreeCheckpointer


# saved in /tmp/nanodo_workdir/{step}/state
def load_params(checkpoint):
    """
    Load params to be used for transformer decoder

    Args:
        checkpoint (string || int): If int, load /tmp/nanodo_workdir/{checkpoint}/state
            If string, load the entire string
    """
    if isinstance(checkpoint, str):
        params = PyTreeCheckpointer().restore(checkpoint)
    elif isinstance(checkpoint, int):
        params = PyTreeCheckpointer().restore(f"/tmp/nanodo_workdir/{checkpoint}/state")
    else:
        print("invalid checkpoint")
        return False
    if 'params' in params:
        model_params = params['params']
        #print("'params' found successfully")
        return params['params']
    else:
        print("'params' not found in checkpoint")


if __name__ == "__main__":
    checkpoint_folder = "/tmp/nanodo_workdir/68000/state"
    params = PyTreeCheckpointer().restore(checkpoint_folder)
    print(params['params'])
    for keys in params['params']:
        print(keys)


    """
    test = load_params(68000)
    test2 = load_params(checkpoint_folder)
    print(len(test))
    print(len(test2))
    """
