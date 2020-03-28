import torch

def vectorize_weights(params):
    return torch.cat([torch.flatten(w.data) for w in params])


def get_state_dict(state_dict, weights_vector):
    i = 0
    state_dict_keys = list(state_dict.keys())
    for key in state_dict_keys:
        old_weights = state_dict[key]
        p_size = torch.numel(old_weights)
        new_weights = weights_vector[i : i + p_size].reshape_as(old_weights)
        state_dict[key] = new_weights
        i += p_size
    return state_dict