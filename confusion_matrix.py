from tensorflow.keras.models import load_model
from load_dataset import val_data

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import numpy as np

# Load model
model = load_model("plant_disease_model.keras")

# Reset generator
val_data.reset()

# Predict all validation images
predictions = model.predict(val_data)

# Convert probabilities to class indices
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_classes = val_data.classes

# Class names
class_names = list(val_data.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)