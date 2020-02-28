
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
from skimage.transform import resize

image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


indices = [14, 26, 32, 49, 52]


#task4b
def plot_first_layer_filter_and_activations():
    plt.figure(figsize=(len(indices)*8, 20))
    for i in range(len(indices)):
        plt.subplot(2, len(indices), i+1)
        plt.title("filter" + str(indices[i]))
        plt.imshow(torch_image_to_numpy(first_conv_layer.weight[indices[i]]))
        plt.subplot(2, len(indices),len(indices) + i + 1)
        plt.title("activation")
        plt.imshow(torch_image_to_numpy(first_conv_layer(image)[0][indices[i]]))
    plt.show()

#task4c
def plot_activations_from_last_layer():
    x = model.conv1(image)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    plt.figure(figsize=(100, 20))
    for i in range(10):
        np_image = torch_image_to_numpy(x[0, i, : , :])
        np_image_upscaled = resize(np_image, (224, 224), order=0)
        plt.subplot(2, 10, i + 1)
        plt.title("activation no.:" + str(i+1))
        plt.imshow(np_image)
        plt.subplot(2, 10, 11 + i)
        plt.title("upsc. " + str(i+1) + " with overlay")
        plt.imshow(np_image_upscaled)
        plt.imshow(torch_image_to_numpy(image[0]), alpha=0.3)
    plt.show()



plot_first_layer_filter_and_activations() #task 4b

plot_activations_from_last_layer() #task 4c
