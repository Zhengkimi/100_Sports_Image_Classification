# 100 Sports Image Classification

## Introduction

Comprising 100 distinct sports categories, this dataset forms the basis of our project. Our objective is to create a CNN model capable of accurately classifying images across these 100 categories. With 13,493 images designated for training and 500 images for validation and testing purposes, each image is standardized to a size of 224x224 pixels and comprises RGB channels.

Data source: [kaggle/100 Sports Image Classification](https://www.kaggle.com/datasets/gpiosenka/sports-classification)

## Model

We explore multiple pre-trained CNN models, such as ResNet, EfficientNet, to leverage their learned representations and adapt them to our specific task by fine-tuning with our dataset.

### Model 1: ResNet50

Res50 is a convolutional neural network architecture that belongs to the ResNet (Residual Network) family. It consists of 50 layers, hence the name "Res50". ResNet was introduced to address the degradation problem in very deep neural networks, where adding more layers led to higher training error. The key innovation in ResNet is the introduction of skip connections, or residual connections, which allow gradients to flow more directly through the network during training, mitigating the vanishing gradient problem. This architecture has proven to be highly effective in various computer vision tasks, including image classification, object detection, and segmentation. Res50 is particularly popular due to its balance between model complexity and performance, making it a widely used choice in many deep learning applications.

**Related Paper**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

**Pretrain Model Source**: [Pytorch](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)

### Model 2: Efficient Net

EfficientNet is a convolutional neural network designed for high performance with fewer parameters and computations. It utilizes a unique scaling method to balance model size and computational cost, making it more resource-efficient compared to other architectures. The series consists of models ranging from EfficientNet-B0 to EfficientNet-B7, with each version progressively increasing in depth, width, and resolution to improve performance. EfficientNet has gained popularity for its state-of-the-art performance on various image recognition tasks, including image classification, object detection, and segmentation.

Our investigation involves testing the performance of EfficientNet models ranging from B0 to B4.

**Related Paper**: Tan, M., & Le, Q. (2021, July). Efficientnetv2: Smaller models and faster training. In International conference on machine learning (pp. 10096-10106). PMLR.

**Pretrain Model Source**: [Github lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

## Training Parameter Setting

1. Image augmentation and normalization: We add HorizontalFlip image to training data set, and normalize the image with mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$ respectively.
2. Training Processing:
    * Gradient Decent Algorithm: Adam
    * Batch Size: $32$
    * Epoch Setting: $100$(Typically, early stopping would occur.)
    * Learning Rate: $0.0005$ in Efficient Net, $0.00005$ in ResNet50
    * Early Stop: When the validation data loss does not decrease for five epochs anymore.

## Performance

|  Model   | Testing Data Accuracy  |
|  :----:  | :----:  |
| ResNet50 | 0.984 |
| EfficientNetB0  | 0.966 |
| EfficientNetB1  | 0.964 |
| EfficientNetB2  | 0.952 |
| EfficientNetB3  | 0.952 |
| EfficientNetB4  | 0.946 |

