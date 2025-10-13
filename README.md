# Traffic Sign Recognition using TensorFlow

This project implements a **Convolutional Neural Network (CNN)** to automatically classify traffic signs from the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset using **TensorFlow** and **Keras**.  
The model is trained to identify **43 different traffic sign categories**, achieving **over 98% accuracy** on the test set.

---

## Project Overview
The goal of this project is to build a deep learning model capable of detecting and classifying traffic signs from real-world images — a key component of intelligent driver-assistance systems (ADAS) and autonomous vehicles.

---

## How It Works
1. **Data Loading**  
   - The dataset is organized into folders from `0` to `42`, each representing a traffic sign class.  
   - Images are resized to `64x64` pixels for consistency.  

2. **Model Architecture**  
   - Four convolutional blocks with **Batch Normalization** and **MaxPooling** layers.  
   - A **Dense layer (512 units)** with **Dropout** for regularization.  
   - A **Softmax output layer** for classifying 43 categories.  

3. **Training**  
   - Optimized using **Adam** optimizer and **categorical cross-entropy loss**.  
   - Achieves around **98–99% accuracy** on the test dataset after 10–15 epochs.  

4. **Prediction**  
   - After training, the model can predict any traffic sign image with a simple command-line interface.  
   - Example:  
     ```
     Model prediction: STOP SIGN (Confidence: 99%)
     ```

---

## Results
| Metric | Score |
|:-------|:------:|
| Training Accuracy | ~99% |
| Test Accuracy | ~98–99% |
| Validation Loss | < 0.05 |

The model demonstrates strong generalization and high reliability on unseen traffic sign images.

---

## Usage

### Train the Model
```bash
python traffic.py gtsrb model.h5
```

### Predict Test Images
```bash
python traffic.py --predict-test-images model.h5 test_images
```

