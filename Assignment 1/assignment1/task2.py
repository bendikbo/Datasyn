import matplotlib.pyplot as plt
import numpy as np

import utils
from task2a import cross_entropy_loss, BinaryModel, pre_process_images

np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: BinaryModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 1]
        model: model of class BinaryModel
    Returns:
        Accuracy (float)
    """

    """
    Task 2c

    Note: Rounding the differences to nearest integer [np.rint(x) returns 0 for x <= 0.5]
    gives a list with zeros representing correct predictions and ones representing wrong predictions.
    """
    outputs = model.forward(X)
    errors = sum(np.rint(np.abs(targets - outputs)))
    return 1 - errors / len(targets)


def train(
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float  # Task 3 hyperparameter. Can be ignored before this.
):
    """
        Function that implements logistic regression through mini-batch
        gradient descent for the given hyperparameters
    """
    global X_train, X_val, X_test
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    model = BinaryModel(l2_reg_lambda)

    #X_train = pre_process_images(X_train)
    #X_test = pre_process_images(X_test)
    #X_val = pre_process_images(X_val)


    early_stop = False

    global_step = 0

    losses = np.array([])


    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            # Select our mini-batch of images / labels
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            ## Task 2b
            # Weights are initialized to zero vector when model object is created
            print(Y_train.shape)
            outputs = model.forward(X_batch)
            model.backward(X_batch, outputs, Y_batch)
            model.w -= learning_rate * model.grad

            # Track training loss continuously
            _train_loss = cross_entropy_loss(Y_batch, outputs)
            train_loss[global_step] = _train_loss
            # Track validation loss / accuracy every time we progress 20% through the dataset
            if global_step % num_steps_per_val == 0:
                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                val_loss[global_step] = _val_loss
                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)
            global_step += 1

        if early_stop:
            print("Epoch: ", epoch)
            break

    return model, train_loss, val_loss, train_accuracy, val_accuracy



# Load dataset
category1, category2 = 2, 3
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_binary_dataset(
    category1, category2, validation_percentage)

X_train = pre_process_images(X_train)
X_test = pre_process_images(X_test)
X_val = pre_process_images(X_val)

# hyperparameters
num_epochs = 500
learning_rate = 0.2
batch_size = 128
l2_reg_lambda = 0
model, train_loss, val_loss, train_accuracy, val_accuracy = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=l2_reg_lambda)

print("Final Train Cross Entropy Loss:",
      cross_entropy_loss(Y_train, model.forward(X_train)))
print("Final Test Entropy Loss:",
      cross_entropy_loss(Y_test, model.forward(X_test)))
print("Final Validation Cross Entropy Loss:",
      cross_entropy_loss(Y_val, model.forward(X_val)))

print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
print("Test accuracy:", calculate_accuracy(X_test, Y_test, model))

# Plot loss
plt.figure(0)
plt.ylim([0., .4])
plt.xlabel("Number of gradient steps")
plt.ylabel("Cross Entropy Loss")
utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.legend()
plt.savefig("binary_train_loss.png")
#plt.show()

# Plot accuracy
plt.figure(1)
plt.ylim([0.93, .99])
plt.xlabel("Number of gradient steps")
plt.ylabel("Accuracy factor")
utils.plot_loss(train_accuracy, "Training Accuracy")
utils.plot_loss(val_accuracy, "Validation Accuracy")
plt.legend()
plt.savefig("binary_train_accuracy.png")


# plot l2reg lambda = [1.0, 0.1, 0.01, 0.001]



_lambda = [1.0, 0.1, 0.01, 0.001, 0 ]

plt.figure(2)
#plt.ylim([0.93, .99])
plt.xlabel("Number of gradient steps")
plt.ylabel("Accuracy factor")

weights = []

for l in _lambda:

    num_epochs = 500
    learning_rate = 0.2
    batch_size = 128
    l2_reg_lambda = l
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        l2_reg_lambda=l2_reg_lambda)
    print(l)
    weights.append(model.w)


    #utils.plot_loss(train_accuracy, "Training Accuracy")
    utils.plot_loss(val_accuracy, "Validation Accuracy: lambda = "+ str(l))
    #utils.plot_loss(val_loss, "Validation Loss " + str(l))
plt.savefig("lambdas.png")

plt.legend()

plt.figure(3)
plt.plot(_lambda, [np.linalg.norm(weights) for weights in weights])
plt.xlabel("lambda")
plt.ylabel("L2 norm")
plt.savefig("weightlength.png")
#Print as image


plt.figure(4)
images = []
for i in range(len(_lambda)):
    image = np.reshape(weights[i][:-1], (28, 28))
    image_min = np.min(image)
    image = (image - image_min) / (np.max(image) - image_min)
    images.append(image)

images = np.concatenate(images, axis=1)
plt.imshow(images)
plt.savefig("weights.png")



plt.show()
