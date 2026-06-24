from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from load_dataset import train_data, val_data

model = Sequential()

model.add(Input(shape=(128,128,3)))

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu'
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)
model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)
model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)
model.add(
    Flatten()
)
model.add(
    Dense(
        128,
        activation='relu'
    )
)
model.add(
    Dense(
        3,
        activation='softmax'
    )
)

model.summary()
print(model.output_shape)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training samples:", train_data.samples)
print("Validation samples:", val_data.samples)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=[early_stop]
)

model.save("plant_disease_model.keras")