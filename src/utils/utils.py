import pickle

import numpy as np
import torch


def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False)).cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices


def pickle_it(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='iso-8859-1')


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def toggle_grad(model, requires_grad: bool):
    """
    Toggles the gradient of the model
    :param model:
    :param requires_grad:
    :return:
    """
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def np_relu(x):
    return np.maximum(0, x)
