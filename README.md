# Intracranial-hemorrhage-detection

# Overview
This project aims to detect and classify intracranial hemorrhage (hemorrhage inside the brain) in CT scan images. The project consists of two main parts:

Classification of Hemorrhage Detection: In this part, a deep learning model is utilized to classify CT scan images and determine whether hemorrhage is present or not. The model employs four convolutional layers and two fully connected layers to achieve accurate classification.

Segmentation of Hemorrhage Localization: The second part focuses on localizing and segmenting the exact regions of hemorrhage in the CT scan images. A masking technique is applied to identify dense areas in the image, as hemorrhage typically occurs in specific spots in the brain. DenseNet architecture is employed to generate blue color overlays on the segmented regions to visually highlight the hemorrhage presence.

# Project Structure

- models/                (Directory for storing trained models)//model is too big around 255Mb 
- src/
  - app.py    (Python script for the hemorrhage classification model)
  - colab_Notebook_Intracranial_hemmorhage.ipynb     (jupyter notebook for the hemorrhage segmentation model)
- README.md              (Project documentation - this file)


# Requirements

Python 3.x
Deep Learning Libraries (e.g., TensorFlow, Keras)
CT scan image dataset (https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data)
Usage

Classification of Hemorrhage Detection:

Prepare your CT scan image dataset and store it in the data/ directory.
Train the classification model using the app.py script. Tweak hyperparameters and model architecture as needed.
After training, save the model in the models/ directory for future use.
Segmentation of Hemorrhage Localization:

Ensure you have the hemorrhage classification model trained in the first part.
Prepare your CT scan image dataset and store it in the data/ directory.
Use the jupyter notebook script to train the segmentation model based on the dense areas identified using masking technology.
Save the segmentation model in the models/ directory, in model.h5 format.

# Conclusion


With the completion of this project, a deep learning-based solution is available to classify and localize intracranial hemorrhage in CT scan images. The classification model can quickly determine the presence of hemorrhage, while the segmentation model accurately highlights the affected regions. The project can contribute significantly to early diagnosis and intervention in intracranial hemorrhage cases, potentially saving lives and improving patient outcomes.

