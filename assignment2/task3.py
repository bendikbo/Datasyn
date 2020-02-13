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
    
    model.ws.clear()
    

#3d
def momentum_gradient_update_step(model: SoftmaxModel, learning_rate: float, prev_moment: np.ndarray, mu: float):
    subtract_ws = np.array([])
    for w in model.ws:
        momentum = prev_moment * mu + (1 - mu) * grad
        np.append(subtract_ws, alpha * momentum)
    np.subtract()







        
    for w in weights:
         momentum = prev_momentum * gamma + (1-gamma)*grad
        w -= alpha * momentum

        momentum = prev_momentum * gAMMa + alpha*grad
        w -= momentum
