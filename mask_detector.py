# Complete Face Mask Detector using Scikit-Learn (No Deep Learning)

# This version uses image features (like pixel values) and trains a basic classifier like SVM.

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# --- Step 1: Load and preprocess the data ---
data = []
labels = []

IMG_SIZE = 100  # Resize all images to 100x100

categories = ["with_mask", "without_mask"]

for category in categories:
    path = os.path.join("dataset", category)
    label = category

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data.append(gray.flatten())  # Convert to 1D array
        labels.append(label)

# --- Step 2: Encode Labels ---
le = LabelEncoder()
labels = le.fit_transform(labels)  # 0 = with_mask, 1 = without_mask

# --- Step 3: Train-Test Split ---
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 4: Train Model ---
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# --- Step 5: Evaluate ---
preds = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, preds, target_names=le.classes_))

# --- Step 6: Save Model and Label Encoder ---
joblib.dump(model, "mask_detector_sklearn.model")
joblib.dump(le, "label_encoder.pkl")

print("\nModel and label encoder saved successfully.")
