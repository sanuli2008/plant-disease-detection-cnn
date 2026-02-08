#  Plant Disease Prediction using Convolutional Neural Networks (CNN)

##  Project Overview

This project focuses on building an **end-to-end Deep Learning system** to automatically detect and classify **plant diseases from leaf images** using a **Convolutional Neural Network (CNN)**. The model is trained on the popular **PlantVillage dataset** and is capable of classifying **38 different plant disease and healthy categories**.

This implementation was developed by following and learning from the YouTube tutorial:

> **Siddhardhan – DL Project 7: Plant Disease Prediction with CNN (End-to-End Deep Learning Project | Docker)**

The project demonstrates the complete deep learning workflow including **data collection, preprocessing, model building, training, evaluation, and prediction**.

---

##  Objectives

* Build a CNN-based image classification model for plant disease detection
* Preprocess and handle a large real-world image dataset
* Achieve high validation accuracy while preventing overfitting
* Create a reusable prediction pipeline for new plant images

---

##  Dataset Information

* **Dataset Name:** PlantVillage Dataset
* **Source:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
* **Total Classes:** 38
* **Image Size:** 256×256 (resized to 224×224 during training)
* **Categories:** Healthy and diseased plant leaves across multiple crops

Only the **color images** were used for this project.

---

##  Technologies & Libraries Used

* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow & Keras
* **Data Handling:** NumPy, OS, JSON
* **Image Processing:** Pillow (PIL), Matplotlib
* **Dataset Handling:** Kaggle API

---

##  Reproducibility

To ensure consistent and reproducible results, random seeds were set for:

* Python `random`
* NumPy
* TensorFlow

---

##  Data Preprocessing

* Dataset extracted from Kaggle ZIP file
* Images rescaled to `[0,1]`
* Images resized to **224×224**
* Dataset split into:

  * **80% Training data**
  * **20% Validation data**
* Used `ImageDataGenerator` for efficient loading

---

##  Model Architecture (CNN)

The CNN model consists of:

* 2 Convolutional layers (ReLU activation)
* Max Pooling layers for downsampling
* Global Average Pooling (to reduce overfitting)
* Fully connected Dense layer
* Dropout layer (0.5)
* Softmax output layer (38 classes)

###  Why Global Average Pooling?

* Reduces the number of trainable parameters
* Helps prevent overfitting
* Improves generalization

---

##  Model Compilation

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Metric:** Accuracy
* **Early Stopping:** Enabled to stop training when validation loss stops improving

---

##  Model Training

* **Epochs:** Up to 20 (Early stopping applied)
* **Batch Size:** 32

###  Final Performance

* **Validation Accuracy:** ~87%
* **Validation Loss:** ~0.40

Training and validation accuracy/loss curves were plotted to monitor learning behavior.

---

##  Prediction System

A prediction pipeline was built to:

* Load a new leaf image
* Resize and normalize the image
* Predict the plant disease class

###  Example Output

```
Predicted Class Name: Potato___Early_blight
```

---

##  Model Saving

The trained model was saved for reuse:

* Format: `.h5` (Keras legacy format)
* Class labels stored in `class_indices.json`

---

##  Project Structure (Suggested)

```
├── dataset/
│   └── plantvillage dataset/
├── notebooks/
│   └── plant_disease_prediction.ipynb
├── model/
│   └── plant_disease_prediction_model.h5
├── class_indices.json
├── README.md
```

---

##  Limitations

* No data augmentation enabled (can be improved)
* Model trained only on color images
* Real-world performance may vary due to lighting and background noise

---

##  Future Enhancements

* Apply data augmentation techniques
* Use Transfer Learning (ResNet, EfficientNet, MobileNet)
* Convert model to `.keras` format
* Deploy using Flask / FastAPI / Docker
* Build a web or mobile application

---

##  Acknowledgements

* **PlantVillage Dataset** – Kaggle
* **Siddhardhan (YouTube)** – for project guidance and explanation
* TensorFlow & Keras Documentation

---

##  License

This project is for **educational purposes only**. Dataset license applies as per Kaggle terms.

---

*If you find this project helpful, feel free to star the repository!*
