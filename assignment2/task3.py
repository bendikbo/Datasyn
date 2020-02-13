import numpy as np
from task2a import SoftmaxModel



#3a
def shuffle_training_data(X: np.ndarray):
    return np.random.permutation(X)


#3b
def better_sigmoid(x):
    return 1.1759*np.tanh(2*x/3)


#3c
def re_init_weights(model: SoftmaxModel):
    prev = model.I
    model.ws.clear()
    for size in model.neurons_per_layer:
        w_shape = (prev, size)
        print("Re-init weights to shape:", w_shape)
        w = np.random.normal(0, 1/np.sqrt(prev), size)
        model.ws.append(w)
        prev = size