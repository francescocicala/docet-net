import torch
import numpy as np

def get_cardinality(arch):
    return sum(map(np.prod, arch))


def format_to_shapes(vector, shapes_list):
    j = 0
    out = []
    for i in range(len(shapes_list)):
        s = np.prod(shapes_list[i])
        out.append(vector[j : j + s].reshape(shapes_list[i]))
        j += s
    return out


def vectorize_weights(params):
    return torch.cat([torch.flatten(w.data) for w in params])


def teach(teacher, net):
    W = vectorize_weights(net.W)
    W = teacher(W)
    W = format_to_shapes(W, net.arch)
    net.set_weights(W)