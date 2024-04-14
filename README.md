# -pneumoniadetection
VGG19 Model Prediction:
Loading Pretrained Model and Image:

A pretrained VGG19 model is loaded from the file model_vgg19.h5.
An image (person1946_bacteria_4874.jpeg) is loaded from the validation directory and resized to (224, 224) pixels.
Image Preprocessing and Prediction:

The image is converted to a NumPy array and expanded to fit the model's input shape.
Preprocessing is applied to the image using the preprocess_input function from VGG16.
The model predicts the class probabilities for the input image.
VGG16 Transfer Learning:
Model Initialization:

A VGG16 model with ImageNet weights is initialized, excluding the top classification layer.
Freezing Layers:

All layers of the VGG16 model are set to non-trainable to retain the pretrained weights.
Custom Classification Layer:

A custom classification layer is added to the VGG16 model, and the entire model structure is summarized.
Model Compilation:

The model is compiled with the categorical cross-entropy loss function, Adam optimizer, and accuracy metric.
Data Augmentation:

Image data generators for training and testing are initialized with rescaling and additional augmentation techniques like shear, zoom, and horizontal flip.
Data Loading:

Training and testing datasets are loaded using the data generators, resized to (224, 224) pixels, and batched with a batch size of 32.
Model Training:

The model is trained using the fit_generator method for 5 epochs with the training and testing datasets.
Performance Visualization:

Training and validation loss and accuracy are plotted using Matplotlib to visualize the model's performance.
Model Saving:

The trained model is saved to a file named model_vgg19.h5.
In summary, the first part of the code loads a pretrained VGG19 model to make predictions on a single image, while the second part demonstrates transfer learning with VGG16 on a custom dataset for image classification.
