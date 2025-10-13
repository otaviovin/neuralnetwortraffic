import os
import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations and arrays
import tensorflow as tf  # Deep learning library (Keras included)
import matplotlib.pyplot as plt  # For plotting images and predictions

# ==========================
# Constants and Parameters
# ==========================
IMG_WIDTH = 64  # Width of input images (increased for higher resolution)
IMG_HEIGHT = 64  # Height of input images
NUM_CATEGORIES = 43  # Total number of traffic sign classes in GTSRB dataset
TEST_SIZE = 0.4  # Fraction of dataset to use as test set
EPOCHS = 15  # Number of training epochs (higher for better learning)

# Mapping of class indices to human-readable traffic sign names
CLASS_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 metric tons", 
    11: "Right-of-way at the next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead", 
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right", 
    37: "Go straight or left", 38: "Keep right", 39: "Keep left", 
    40: "Roundabout mandatory", 41: "End of no passing", 
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# ==========================
# Function: Load Dataset
# ==========================
def load_data(data_dir):
    """
    Load images and corresponding labels from the dataset directory.
    Assumes dataset folders are named 0, 1, ..., NUM_CATEGORIES-1.

    Args:
        data_dir (str): Path to the dataset directory.

    Returns:
        images (np.array): Array of image data (height x width x 3).
        labels (np.array): Array of integer class labels.
    """
    images, labels = [], []

    # Loop through each category folder
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        if not os.path.exists(category_path):
            continue  # Skip if category folder does not exist

        # Loop through each image file in the category folder
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path)  # Read image with OpenCV
            if img is None:
                continue  # Skip if image cannot be read

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize to standard size
            images.append(img)
            labels.append(category)  # Assign label based on folder name

    return np.array(images), np.array(labels)

# ==========================
# Function: Build CNN Model
# ==========================
def get_model():
    """
    Create and compile a convolutional neural network (CNN) for traffic sign classification.

    Returns:
        model (tf.keras.Model): Compiled Keras CNN model ready for training.
    """
    model = tf.keras.models.Sequential([
        # 1st Convolutional Block
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.BatchNormalization(),  # Normalizes feature maps to speed up convergence
        tf.keras.layers.MaxPooling2D(2,2),  # Reduce spatial size by 2x

        # 2nd Convolutional Block
        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        # 3rd Convolutional Block
        tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        # 4th Convolutional Block (Added more depth for better feature extraction)
        tf.keras.layers.Conv2D(256, (3,3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        # Fully Connected Layers
        tf.keras.layers.Flatten(),  # Flatten 3D feature maps to 1D vector
        tf.keras.layers.Dense(512, activation="relu"),  # Dense layer for learning complex patterns
        tf.keras.layers.Dropout(0.5),  # Dropout to reduce overfitting
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")  # Output layer with softmax
    ])

    # Compile model with optimizer, loss, and evaluation metrics
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# ==========================
# Function: Train Model
# ==========================
def train_model(data_dir, model_file):
    """
    Train the CNN model on the GTSRB dataset and save the trained model.

    Args:
        data_dir (str): Path to dataset folder.
        model_file (str): Filename to save the trained model (.h5 format).
    """
    from sklearn.model_selection import train_test_split

    # Load dataset
    images, labels = load_data(data_dir)
    labels_cat = tf.keras.utils.to_categorical(labels)  # Convert labels to one-hot vectors

    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(images, labels_cat, test_size=TEST_SIZE)

    # Build the CNN model
    model = get_model()

    # Normalize pixel values to [0,1] and train the model
    model.fit(x_train/255.0, y_train, epochs=EPOCHS, validation_data=(x_test/255.0, y_test))

    # Save the trained model to file
    model.save(model_file)
    print(f"Model saved to {model_file}")

# ==========================
# Function: Predict Traffic Signs
# ==========================
def predict_test_images(model_file, test_dir):
    """
    Predict traffic sign categories for PNG images in a folder and display results.

    Args:
        model_file (str): Path to the trained CNN model (.h5 file).
        test_dir (str): Folder containing PNG test images.
    """
    # Load the trained model
    model = tf.keras.models.load_model(model_file)

    # Loop through all PNG images in the test folder
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(".png"):
            img_path = os.path.join(test_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Resize and normalize the image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)).astype("float32") / 255.0

            # Predict the class probabilities
            pred = model.predict(np.expand_dims(img, axis=0))
            class_idx = np.argmax(pred, axis=1)[0]  # Get index of highest probability
            confidence = np.max(pred) * 100  # Confidence score
            sign_name = CLASS_NAMES.get(class_idx, "Unknown")  # Map index to traffic sign name

            # Print result to console
            print(f"{filename}: {sign_name} ({confidence:.2f}%)")

            # Display image with prediction
            plt.imshow(cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.title(f"{sign_name} ({confidence:.2f}%)")
            plt.axis("off")
            plt.show()

# ==========================
# Main Entry Point
# ==========================
if __name__ == "__main__":
    import sys
    # Prediction mode: python traffic.py --predict-test-images model.h5 test_images
    if len(sys.argv) >= 3 and sys.argv[1] == "--predict-test-images":
        predict_test_images(sys.argv[2], sys.argv[3])
    # Training mode: python traffic.py gtsrb model.h5
    elif len(sys.argv) >= 3:
        train_model(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("Train: python traffic.py gtsrb model.h5")
        print("Predict: python traffic.py --predict-test-images model.h5 test_images")