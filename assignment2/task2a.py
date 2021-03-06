import numpy as np
import utils
import typing
np.random.seed(1)


def improved_cig(x:np.ndarray):
    return 2.28787/(1+np.cosh(4*x/3))

def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """

    X = X-33.31#find_mean(X)
    #print(find_mean(X))
    X = X/78.57#find_standard_deviation(X)
    #print(find_standard_deviation(X))
    X = np.insert(X,-1,1,axis = 1)


    assert X.shape[1] == 785,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    return X

def find_mean(X: np.ndarray):
    return np.mean(X)

def find_standard_deviation(X: np.ndarray):
    return np.std(X)


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    size = outputs.size

    cross_entr_err = np.sum(np.multiply(targets, np.log(outputs)))#+np.dot((1-targets).T, np.log(1-outputs)))
    cross_entr_err = -cross_entr_err/targets.shape[0]

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return cross_entr_err


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

        #improvements
        self.better_sigmoid = False

        #chaches
        self.activations = []
        self.forwards = []
        # Initialize the weights
        self.ws = []
        prev = self.I
        self.moment = []
        if use_improved_weight_init:
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                momentum = np.zeros(w_shape)
                self.moment.append(momentum)
                print("Initializing weight to shape:", w_shape)
                w = np.random.normal(0, 1/np.sqrt(prev),w_shape)
                self.ws.append(w)
                prev = size
        else:
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                momentum = np.zeros(w_shape)
                self.moment.append(momentum)
                print("Initializing weight to shape:", w_shape)
                w = np.random.uniform(-1,1,w_shape)#np.zeros(w_shape)
                self.ws.append(w)
                prev = size
        self.grads = [None for i in range(len(self.ws))]
        prev = self.I


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        forwards = []
        activations = []


        prev = X
        activations.append(prev)
        for i in range(len(self.ws)-1):
            z = np.dot(prev,self.ws[i])
            #sigmoid
            prev = 1/(1+np.exp(-z))
            forwards.append(z)
            activations.append(prev)

        z = np.dot(prev,self.ws[-1])
        forwards.append(z)

        #softmax
        prev = np.divide(np.exp(z),np.sum(np.exp(z),axis = 1).reshape(-1,1))

        #exponent = np.exp(curr)
        #sum_exponent = sum(exponent)

        #prev = exponent/sum_exponent
        self.forwards = forwards
        self.activations = activations

        return prev
    
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

        activations = self.activations
        forwards = self.forwards

        #zj = np.dot(X, self.ws[0])
        #aj = 1/(1 + np.exp(-zj))

        for i in range(len(self.ws)):
            #last layer
            if i == 0:
                del_k = - (targets - outputs)
                self.grads.append(np.dot(del_k.T, activations[-1]).T/X.shape[0])

            else:
                if self.use_improved_sigmoid:
                    der_sig = improved_cig(forwards[-i-1])
                    del_j = np.dot(del_k, self.ws[-i].T) * der_sig
                    grad_w_ij = np.dot(activations[-i-1].T, del_j)
                    self.grads.insert(0, grad_w_ij/X.shape[0])
                    del_k = del_j
                else:
                    #all hidden layer
                    der_sig = (1/(1 + np.exp(-forwards[-i-1])))*(1 - (1/(1 + np.exp(-forwards[-i-1]))))
                    del_j = np.dot(del_k, self.ws[-i].T)*der_sig
                    grad_w_ij = np.dot(activations[-i-1].T,del_j)
                    self.grads.insert(0, grad_w_ij/X.shape[0]) #insert in the beginning of list
                    del_k = del_j




        #grad_w_kj = (np.dot(del_k.T, activations[-1])).T #outputlayer

        #der_sig = (1/(1 + np.exp(-zj)))*(1 - (1/(1 + np.exp(-zj))))

        #del_j = np.dot(self.ws[1], del_k.T) * der_sig.T
        #grad_w_ij = np.dot(del_j, X).T

        #self.grads = [grad_w_ij.T/X.shape[0], grad_w_kj.T/X.shape[0]]
        #self.grads = [grad_w_ij/X.shape[0], grad_w_kj/X.shape[0]]

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
    encoded = np.zeros((Y.shape[0], num_classes))
    for val in range(Y.shape[0]):
        encoded[val, Y[val, 0]] = 1
    return encoded


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
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
