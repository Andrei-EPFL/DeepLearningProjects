from torch import empty, manual_seed, set_grad_enabled, set_default_dtype, set_printoptions
import math
import time

import torch
import dl

set_printoptions(precision=30)
set_default_dtype(torch.float32)
set_grad_enabled(False)

def generate_disc_set(nb, one_hot_encode=True):
    ''' 
        Generate the data set:
        All points inside a radius of sqrt( 1 / (2 * PI) ) have
        the label equal to 1 and all outside points have the
        label equal to 0. 
    '''
    data = empty(size=(nb, 2)).uniform_(0, 1)
    radius = (data - 0.5).pow(2).sum(axis=1)
    labels = (radius < 1./(2 * math.pi)).long()
    if one_hot_encode:
        out = empty(size=(data.shape[0], 2)).fill_(0).float()
        out[~labels.bool(),0] = 1
        out[labels.bool(),1] = 1
        return data, out, labels
    else:
        return data, labels

if __name__ == '__main__':
    manual_seed(42)

    train_input, train_target, train_labels = generate_disc_set(1000, one_hot_encode=True)
    validation_input, validation_target, validation_labels = generate_disc_set(1000, one_hot_encode=True)
    test_input, test_target, test_labels = generate_disc_set(1000, one_hot_encode=True)

    torch.save(train_input, "./data/train_input_float32_S42.pt")
    torch.save(train_target, "./data/train_target_float32_S42.pt")
    torch.save(train_labels, "./data/train_labels_float32_S42.pt")

    torch.save(validation_input, "./data/validation_input_float32_S42.pt")
    torch.save(validation_target, "./data/validation_target_float32_S42.pt")
    torch.save(validation_labels, "./data/validation_labels_float32_S42.pt")
    
    torch.save(test_input, "./data/test_input_float32_S42.pt")
    torch.save(test_target, "./data/test_target_float32_S42.pt")
    torch.save(test_labels, "./data/test_labels_float32_S42.pt")
    