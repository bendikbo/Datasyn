import numpy as np
import utils
import typing
np.random.seed(1)

def sigmoid(Z):
    return((1/(1+np.exp(-Z))))

def sigmoid_prime(Z):
    temp_calc = sigmoid(Z)
    return(np.multiply(temp_calc, 1-temp_calc))

def improved_sigmoid(Z):
    return(1.7159*np.tanh(2*Z/3))

def improved_sigmoid_prime(Z):
    return(1.7159*(2/3)*(np.cosh((2/3)*Z)**-2))

def pre_process_images(X: np.ndarray, mean_X_train = 33.34, std_X_train = 78.59):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"

    return np.append((X-mean_X_train)/std_X_train, np.ones((X.shape[0],1)), axis=1)


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    return (-1/(targets.shape[0]))*np.sum(np.sum(np.multiply(targets, np.log(outputs))))
    # raise NotImplementedError


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if use_improved_weight_init:
                w = np.random.normal(loc=0.0, scale=w_shape[0]**-0.5 , size=w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

        self.forward_cache = []
        self.activation_cache = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # We we assume we use sigmoid for all but the last layer where we have
        # to apply the softmax activation. 
        forward_cache = [] # stores the activations of the hidden layers
        activation_cache = []
        a = X
        activation_cache.append(a)
        for ind, w in enumerate(self.ws):
            z = np.dot(a, w)
            forward_cache.append(z)
            if ind < len(self.ws)-1: # sigmoid activation function for everything but the ultimate layer
                if self.use_improved_sigmoid:
                    a = improved_sigmoid(-a.dot(w))
                else:
                    a = sigmoid(-a.dot(w))

                activation_cache.append(a)

        self.forward_cache = forward_cache
        self.activation_cache = activation_cache
        return np.divide(np.exp(z),np.sum(np.exp(z), axis=1).reshape(-1,1))

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        
        self.grads = []
        
        # appending the activations
        forward_cache = self.forward_cache
        activation_cache = self.activation_cache
        # we use the activations from the previous layer when backpropagating 
        # the error.
        for ind in range(len(self.ws)):
            if ind == 0: # calculating the error and grads for the last layer
                error_prev = -(targets - outputs)

            else:
                if self.use_improved_sigmoid:
                    error_prev = -np.multiply(improved_sigmoid_prime(forward_cache[-ind-1]), error_prev.dot(self.ws[-ind].T))
                else:
                    error_prev = -np.multiply(sigmoid_prime(forward_cache[-ind-1]), error_prev.dot(self.ws[-ind].T))
            
            self.grads.insert(0, (activation_cache[-ind-1].T.dot(error_prev))/X.shape[0])    



        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

            

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # raise NotImplementedError
    y_encoded = np.zeros((Y.shape[0], num_classes))
    for ind, elem in enumerate(Y):
        y_encoded[ind, elem] = 1
    return y_encoded


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)

    # This would not be my preferred way of doing this, but anyway. We calculate the
    # mean and std of the train images outside the function to have access to them
    # as variables in the global scope. A better implementation would perhaps define 
    # a preprocess class an fit, it to the train set so that class contains the variables.
    # This is what is done in the sklearn preprocessing package.

    mean_X_train = np.mean(X_train)
    std_X_train = np.std(X_train) # ddof = 0 or ddof =1

    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)
    logits = model.forward(X_train)
    np.testing.assert_almost_equal(
        logits.mean(), 1/10,
        err_msg="Since the weights are all 0's, the softmax activation should be 1/10")

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
