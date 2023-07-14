# CIFAR-10 Dataset PyTorch implementation using Custom Resnet Model
The target is to achieve 90% accuracy using custom resnet model in just 24 epochs
## Basics
The CIFAR10 dataset is a popular benchmark dataset for image classification tasks. In this tutorial, we present an EfficientNet architecture tailored for the CIFAR10 dataset. The architecture consists of multiple layers and incorporates the concepts of residual blocks and one cycle policy for optimized performance.

## Code organization

Code organization in this project is structured into five files.

src

 &nbsp;&nbsp;&nbsp;&nbsp; `models.py` 

 &nbsp;&nbsp;&nbsp;&nbsp; `utils.py`

 &nbsp;&nbsp;&nbsp;&nbsp; `dataset.py`

 &nbsp;&nbsp;&nbsp;&nbsp; `train.py`

 &nbsp;&nbsp;&nbsp;&nbsp; `test.py`

`S10.ipynb`

The file **"models.py"** houses the class that defines the model structure. Within **"utils.py"** you'll find code responsible for generating performance graphs. **"dataset.py"** contains the necessary code for loading the cifar10 dataset. 
 **"train.py"** and **"test.py"** contains code for training, testing. Finally, **"S6.ipynb"** acts as the notebook where the actual execution and experimentation take place. 
 
## Architecture Overview
The EfficientNet architecture for CIFAR10 is designed with the following key components:

- **PrepLayer**: This layer includes a convolutional layer with a 3x3 kernel, stride of 1, and padding of 1, followed by batch normalization (BN) and rectified linear unit (ReLU) activation. It consists of **64k** filters.

- **Layer 1**: This layer includes a convolutional layer with a 3x3 kernel, stride of 1, and padding of 1, followed by max pooling, batch normalization, and ReLU activation. It consists of **128k** filters.

  Residual Block (R1): This block incorporates two convolutional layers with batch normalization and ReLU activation. The output from Layer 1 is added to the output of the residual block.

 - **Layer 2**: This layer consists of a convolutional layer with a 3x3 kernel and **256k** filters, followed by max pooling, batch normalization, and ReLU activation.

- **Layer 3**: Similar to Layer 1, Layer 3 includes a convolutional layer, max pooling, batch normalization, and ReLU activation. It consists of **512k** filters.

    Residual Block (R2): This block is similar to R1 and follows Layer 3. The output from Layer 3 is added to the output of R2.

- **Max Pooling**: A max pooling layer with a **kernel size of 4** is applied to the output of R2.

- **Fully Connected (FC) Layer**: The output from the previous layers is flattened and passed through a fully connected layer followed by a softmax activation function for classification.

 ![image](https://github.com/Paurnima-Chavan/cifar-s10/assets/25608455/fdd1dbf0-1b12-4b18-976d-5a2bd1c0acf0)


### **Transformations**: 

The **albumentations** library is a popular image augmentation library in Python that provides a wide range of transformations to preprocess and augment images.
The architecture applies data augmentation techniques such as random cropping of 32x32 (after padding of 4) and flipping the images horizontally. Additionally, cutout with a size of 8x8 is applied using albumentations.

## Training Configuration
The training process for this model architecture follows the **one cycle policy**. The main idea behind the One Cycle Policy is to start with a low learning rate, gradually increase it to a maximum value, and then decrease it again.
The key training parameters are as follows:
- Total Epochs: The model is trained for a total of 24 epochs.
- Learning Rate (LR): The minimum and maximum learning rates (LRMIN and LRMAX) are determined using a learning rate finder method.
- No Annihilation: No learning rate annihilation is applied during the training process.
- Batch Size: The model is trained using a batch size of 512.
- Optimization Algorithm: ADAM optimizer is used for updating the model weights.
- Loss Function: Cross-Entropy Loss is employed as the loss function for training.

![image](https://github.com/Paurnima-Chavan/cifar-s10/assets/25608455/558dc065-e36e-4230-b643-fd8719d55dd8)

### Training Log Summary for 24 Epochs
The training log summary displays the performance metrics for the model over 24 training epochs. The metrics include training loss, training accuracy, and test accuracy.

![image](https://github.com/Paurnima-Chavan/cifar-s10/assets/25608455/65f4b3d6-e8ad-4ab6-b05d-5486168d11c7)

## Conclusion
The model architecture for CIFAR10 combines various architectural components, including convolutional layers, residual blocks, max pooling, and fully connected layers, to achieve efficient and accurate image classification. The application of the one cycle policy and data augmentation techniques further enhances the model's performance. By leveraging this architecture, we can achieve notable results in CIFAR10 classification tasks
