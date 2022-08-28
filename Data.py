import torch

def normal_batch (size, batch_size, mean=0, std=1) :
# with torch.no_grad():
    return torch.normal(mean, std, size=(batch_size, size, size))

def flip (input_batch) :
# with torch.no_grad():
    return torch.swapaxes(-input_batch, 1, 2)

def flip_cat (input_batch) :
    return torch.cat(input_batch, flip(input_batch), dim=0)

def normal_batch_flip_cat (size, batch_size, mean=0, std=1) :
    return flip_cat(normal_batch(size, batch_size, mean=0, std=1))

def first_half (input_batch) :
    return input_batch[ : input_batch.shape[0]//2]

def second_half (input_batch) :
    return input_batch[input_batch.shape[0]//2 : ]






if __name__ == '__main__':
    
    batch_size = 2**10

    for _ in range(10**6):
        x = normal_batch_flip_cat(3, batch_size)

    