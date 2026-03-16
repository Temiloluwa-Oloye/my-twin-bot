# Fruit Classification using Transfer Learning

A deep learning project that classifies fruit images using transfer learning with the VGG16 architecture. This project demonstrates the application of convolutional neural networks and transfer learning techniques for image classification tasks.

## Problem Statement

The goal of this project is to develop an accurate and efficient fruit image classification system. Traditional image classification approaches often require large datasets and extensive computational resources. This project addresses these challenges by leveraging transfer learning, which allows us to utilize pre-trained models trained on large datasets (like ImageNet) and adapt them to our specific fruit classification task with limited data and computational resources.

The classifier needs to accurately identify and categorize different types of fruits from images, which has applications in automated sorting systems, quality control in food processing, and agricultural automation.

## Dataset Description

The project uses the Fruits-360 dataset, which contains images of various fruits organized into training, validation, and test sets. The dataset includes:

- **Training set**: 6,231 images across 24 fruit classes
- **Validation set**: 3,114 images across 24 fruit classes
- **Test set**: 3,110 images across 24 fruit classes

The dataset covers a diverse range of fruit categories including various apple varieties, pears, cucumbers, zucchinis, carrots, eggplants, and cabbages. Each image is preprocessed to a standardized size of 64x64 pixels for model input.

The dataset is organized in a directory structure where each fruit class has its own subdirectory, making it suitable for use with Keras's `flow_from_directory` method.

## Methodology

### Transfer Learning Approach

The project employs transfer learning using the VGG16 architecture, a deep convolutional neural network pre-trained on the ImageNet dataset. This approach provides several advantages:

1. **Leverages pre-trained features**: The VGG16 model has learned rich feature representations from millions of images
2. **Reduced training time**: Requires less training time compared to training from scratch
3. **Better performance with limited data**: Achieves good results even with smaller datasets

### Model Architecture

1. **Base Model**: VGG16 pre-trained on ImageNet (frozen during initial training)
2. **Custom Layers**:
   - Global Average Pooling 2D layer
   - Dense layer (256 units, ReLU activation)
   - Batch Normalization layer
   - Dropout layer (0.3 rate) for regularization
   - Output layer (24 units, softmax activation) for multi-class classification

### Training Strategy

The training process follows a two-phase approach:

1. **Initial Training Phase**:
   - Freeze all VGG16 base layers
   - Train only the custom classification layers
   - Use Adam optimizer with default learning rate
   - Apply data augmentation (rotation, shifting, zooming, flipping)
   - Implement callbacks: ReduceLROnPlateau and EarlyStopping

2. **Fine-Tuning Phase**:
   - Unfreeze the last 5 layers of VGG16
   - Keep BatchNormalization layers frozen
   - Use lower learning rate (1e-5) for gradual weight adjustments
   - Continue training with the same augmentation and callbacks

### Data Augmentation

To improve model generalization and prevent overfitting, the following augmentation techniques are applied to training data:

- Rotation: ±20 degrees
- Width/Height shifts: 10-20% range
- Shear transformation: 20% range
- Zoom: 20% range
- Horizontal flipping

## Results

The model achieves strong performance on the test set:

- **Test Accuracy**: ~88% (0.88)
- **Test Loss**: ~0.40

The training history shows:
- Steady improvement in both training and validation accuracy
- Effective fine-tuning that further improves model performance
- Good generalization with minimal overfitting

The model successfully classifies various fruit types with high confidence, demonstrating the effectiveness of the transfer learning approach for this classification task.

## Conclusion

This project successfully demonstrates the application of transfer learning for fruit image classification. By leveraging the pre-trained VGG16 model, we achieved high classification accuracy (~88%) with relatively limited computational resources and training time.

Key takeaways:

1. **Transfer learning is highly effective**: Using pre-trained models significantly improves performance compared to training from scratch
2. **Fine-tuning enhances performance**: Unfreezing and fine-tuning specific layers allows the model to adapt better to the target domain
3. **Data augmentation is crucial**: Proper augmentation techniques help improve model generalization
4. **Callbacks optimize training**: Learning rate scheduling and early stopping help achieve better results efficiently

The model can be further improved by:
- Training for more epochs
- Experimenting with different architectures (ResNet, EfficientNet)
- Using larger input image sizes
- Implementing ensemble methods
- Collecting more diverse training data

This project serves as a solid foundation for real-world fruit classification applications and demonstrates practical deep learning techniques for computer vision tasks.
