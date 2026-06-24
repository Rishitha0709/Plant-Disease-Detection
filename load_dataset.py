from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_path =  r"C:\Users\sandh\OneDrive\Desktop\Plant_disease\dataset"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

images, labels = next(train_data)

print(images.shape)
print(labels.shape)
print(train_data.class_indices)

import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
plt.imshow(images[0])
plt.axis("off")
plt.show(block=True)