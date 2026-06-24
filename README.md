# Plant-Disease-Detection
CNN-based Plant Disease Detection system trained on the PlantVillage Dataset for classifying potato leaf diseases.

## Overview

Plant Disease Detection is a deep learning-based system developed to identify diseases in potato leaves using image classification. The model is trained on the PlantVillage Dataset and can classify leaf images into multiple disease categories, helping in early disease detection and crop management.

## Dataset

* PlantVillage Dataset
* Classes:

  * Potato___Early_blight
  * Potato___Late_blight
  * Potato___healthy

## Features

* Image-based plant disease classification
* CNN-based deep learning model
* Automated disease prediction from leaf images
* Model evaluation using confusion matrix and classification metrics
* Support for healthy and diseased leaf detection

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Project Structure

* train_model.py – Model training
* predict.py – Disease prediction on new images
* load_dataset.py – Dataset loading and preprocessing
* confusion_matrix.py – Model evaluation
* check_dataset.py – Dataset inspection utilities

## Results

* Achieved approximately 92% validation accuracy on the PlantVillage Dataset.
* Successfully classified potato leaf diseases including Early Blight, Late Blight, and Healthy leaves.

## Sample Prediction

Input: Potato leaf image

Output:

* Potato___Early_blight
* Potato___Late_blight
* Potato___healthy

## Future Enhancements

* Support for additional crop species
* Web-based deployment
* Real-time disease detection using mobile devices

## Note

The PlantVillage dataset and trained model files are not included in this repository due to size limitations.
