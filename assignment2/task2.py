import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
np.random.seed(0)
import random


def equal_random_permutation(X: np.ndarray, Y: np.ndarray):
    tuple_x_y = list(zip(X, Y))
    random.shuffle(tuple_x_y)
    new_x, new_y = zip(*tuple_x_y)
    return np.array(new_x), np.array(new_y)






def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    raw_results = model.forward(X)
    accuracy = 0
    for i in range(len(raw_results)):
        accuracy += targets[i, np.argmax(raw_results[i])]/len(raw_results)
    return accuracy

def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            #Select our mini-batch of images / labels
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]


            #task2c
            #randomize weights for hidden layers

            outputs = model.forward(X_batch)
            model.backward(X_batch, outputs, Y_batch)


            for i in range(len(model.ws)):
                if use_momentum:
                    model.moment[i] *= momentum_gamma
                    model.moment[i] += model.grads[i]*(1-momentum_gamma)
                    model.ws[i] -= learning_rate * model.moment[i]
                else:
                    model.ws[i] -= model.grads[i]
               


            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            if (global_step % num_steps_per_val) == 0:
                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                val_loss[global_step] = _val_loss
                _train_loss = cross_entropy_loss(Y_train, model.forward(X_train))
                train_loss[global_step] = _train_loss
                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1
            #3a
        if use_shuffle:
            X_train, Y_train = equal_random_permutation(X_train, Y_train)

    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)

    X_train = pre_process_images(X_train)
    X_test = pre_process_images(X_test)
    X_val = pre_process_images(X_val)

    Y_train = one_hot_encode(Y_train,10)
    Y_val = one_hot_encode(Y_val,10)
    Y_test = one_hot_encode(Y_test,10)

    # Hyperparameters
    num_epochs = 20
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [64, 64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    # Settings for task 3. Keep all to false for task 2.
    use_shuffle = True
    use_improved_sigmoid = True
    use_improved_weight_init = False
    use_momentum = False

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        model,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        momentum_gamma=momentum_gamma)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model))

    # Plot loss
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.ylim([0,1])
    #plt.ylim([0.1, .5])
    utils.plot_loss(train_loss, "Training Loss")
    utils.plot_loss(val_loss, "Validation Loss")
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    # Plot accuracy
    plt.ylim([0.9, 1.0])
    utils.plot_loss(train_accuracy, "Training Accuracy")
    utils.plot_loss(val_accuracy, "Validation Accuracy")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.savefig("softmax_train_graph_task4d.png")
    plt.show()
