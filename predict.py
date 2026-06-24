from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model("plant_disease_model.keras")

# Image path
img_path = r"C:\Users\sandh\OneDrive\Desktop\Plant_disease\new_leaf.jpg"

# Load image
img = image.load_img(
    img_path,
    target_size=(128,128)
)

# Convert to array
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

predicted_class = np.argmax(prediction)

print("Prediction:", classes[predicted_class])

print("\nProbabilities:")
for i in range(len(classes)):
    print(f"{classes[i]}: {prediction[0][i]*100:.2f}%")