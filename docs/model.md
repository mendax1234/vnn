# CNN Model

VNN is validated using a custom CNN designed for CIFAR-10 image classification. The model follows a compact convolutional architecture composed of stacked convolutional blocks, max-pooling layers, global average pooling, and a final fully connected classifier.

## Model Architecture

The network uses repeated **Conv-BatchNorm-ReLU** blocks for feature extraction. This pattern improves training stability through batch normalization and introduces non-linearity through ReLU activation after each convolution. As the network goes deeper, the number of channels increases progressively, allowing later layers to learn richer and more abstract image features.

The model adopts several design patterns commonly used in modern CNN architectures:

* **Deep feature extraction** through multiple stacked convolutional layers
* **Progressive channel expansion** from 16 to 32 to 64 feature channels
* **Spatial downsampling** using max-pooling layers
* **Global average pooling** before classification to reduce parameter count
* **A lightweight fully connected layer** for the final 10-class CIFAR-10 output

The overall architecture is summarized below.

| Layer Type | Output Dimensions | Number of Operations |
| ---------- | ----------------: | -------------------: |
| Conv2D     |      16 × 32 × 32 |              475,136 |
| MaxPool    |      16 × 16 × 16 |                3,072 |
| Conv2D     |      32 × 16 × 16 |            1,196,032 |
| MaxPool    |        32 × 8 × 8 |                8,192 |
| Conv2D     |        64 × 8 × 8 |            1,187,840 |
| Conv2D     |        64 × 8 × 8 |            2,367,488 |
| AvgPool    |                64 |                4,096 |
| FC         |                10 |                  650 |

!!! note

    Each Conv2D layer includes ReLU activation.

## Training

The model is trained in PyTorch for **100 epochs** using the **AdamW** optimizer. A cosine-decay learning rate schedule is used to gradually reduce the learning rate during training, helping the model converge more smoothly toward the end of training.

Before training, input images are normalized. Standard data augmentation techniques, including random cropping and horizontal flipping, are applied to improve generalization and make the model more robust to small variations in the input images.

## Quantization

After training, the learned weights are quantized to a **16-bit fixed-point format**, represented as **q6.10**. This format uses 6 integer bits and 10 fractional bits, providing a balance between numerical precision and hardware efficiency.

Fixed-point quantization reduces the cost of arithmetic operations compared with floating-point computation, making the model more suitable for RTL implementation on FPGA hardware. In VNN, this quantized representation is used during hardware inference.

<!-- Add roofline model analysis here -->
