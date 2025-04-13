import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Argument parsing
parser = argparse.ArgumentParser(description="Experiment 2: Fine-tune Stanford Dogs model on Cats vs Dogs dataset")
parser.add_argument("log_dir", type=str, help="Directory to save logs")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
args = parser.parse_args()

# Dataset Path
dataset_path = "/datasets/PetImages"

# Load the saved model from Experiment 1
saved_model_path = "Saved_model/stanford_dogs_model_final.keras"
model = tf.keras.models.load_model(saved_model_path)

image_size = (180, 180)
batch_size = 64

# Load the Cats vs Dogs dataset
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training", 
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Data augmentation layers
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

# Apply data augmentation function
def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

# Prepare the datasets with data augmentation
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

model.layers.pop()
x = layers.Dense(1, activation="sigmoid", name="output_dense")(model.output)

model = tf.keras.Model(inputs=model.input, outputs=x)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("fine_tuned_model.keras"),
    tf.keras.callbacks.TensorBoard(log_dir=args.log_dir),
]

model.fit(
    train_ds,
    epochs=args.epochs,
    validation_data=val_ds,
    callbacks=callbacks,
)

# Save
model.save("fine_tuned_model_final.keras")
