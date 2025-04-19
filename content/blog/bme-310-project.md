# Brain Tumor Classification Using Deep Learning: A BME 310 Project

Hello everyone! This post showcases a project my partner, Ifreet Tahia, and I completed for our BME 310 lab, "Communicating Protocols for Biomedical Instruments Sessional". This lab introduced us to Python programming and essential libraries for machine learning like NumPy, Pandas, Matplotlib, Scikit-learn, and SciPy.

## Project Overview

Our project focused on **classifying brain tumors using Magnetic Resonance Imaging (MRI) scans**. We implemented a deep learning approach based on the research paper **"BrainMRNet: Brain Tumor Detection using Magnetic Resonance Images with a Novel Convolutional Neural Network Model"**.

The goal was to build a model that could automatically distinguish between MRI images showing brain tumors and those that are normal. The BrainMRNet model, as described in the paper, is a Convolutional Neural Network (CNN) architecture specifically designed for this task. It utilizes techniques like attention modules to focus on relevant areas of the image, residual blocks to improve training, and a hypercolumn technique to retain features from multiple layers for better classification.

## Why Is This Important?

Deep learning models, like the one we implemented, offer a promising avenue for improving the speed and accuracy of brain tumor detection from MRI scans. Computer-assisted diagnosis tools can help medical professionals identify tumors earlier, potentially leading to better patient outcomes. Our project explores how these advanced machine learning techniques can be applied to this critical area of biomedical engineering.

## Skills We Learned

This project was a fantastic learning experience. Key skills we developed include:

* **Python Programming**: Strengthening our foundational programming skills in Python, the language used for implementation.
* **Deep Learning Concepts**: Understanding and applying concepts of Convolutional Neural Networks (CNNs), including architecture design (attention modules, residual blocks, hypercolumn technique).
* **Machine Learning Libraries**: Gaining hands-on experience with essential Python libraries for data science and machine learning.
* **Image Processing**: Learning techniques for handling and preparing medical images (MRIs) for model training, potentially including preprocessing and data augmentation.
* **Model Implementation**: Translating a research paper's methodology into functional code.
* **Model Evaluation**: Assessing the performance of the deep learning model using appropriate metrics (like accuracy, auc, sensitivity, specificity).
* **Tensor Manipulation**: Manipulating tensor shapes to fit with different layers of the model.
* **Code Reproducibility**: Extra effort was given to make the code reproducible by avoiding randomness using SEED

## Libraries We Used

To bring this project to life, we utilized several powerful Python libraries:

* **PyTorch**: The model and training loop were implemented using PyTorch library.
* **PIL**: Pillow library was used for image-file reading.
* **pathlib**: Pathlib library was used for handling paths in operating system independent manner. 

## Result

| Model        | Accuracy | AUC    | Sensitivity | Specificity | Avg Precision | Avg Recall |
|--------------|----------|--------|-------------|-------------|----------------|------------|
| VGG16        | 0.7800   | 0.8050 | 0.8667      | 0.6500      | 0.7763         | 0.7583     |
| ResNet50     | 0.7600   | 0.9417 | 1.0000      | 0.4000      | 0.8571         | 0.7000     |
| DenseNet121  | 0.9400   | 0.9833 | 0.9667      | 0.9000      | 0.9414         | 0.9333     |
| BrainMRNet   | 0.8800   | 0.9167 | 0.9333      | 0.8000      | 0.8819         | 0.8667     |

The BrainMRNet model performed better than VGG16 and ResNet50. But BrainMRNet model failed to beat the DenseNet121 model. But it's okay. Because the BrainMrNet model has only 0.5 million parameters whereas DenseNet121 model has 8 million parameters.

## Code

The full code is available in my [Github repo](https://github.com/Marshal-Ashif-Shawkat/bme-310-project)
