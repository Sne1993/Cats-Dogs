import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Argument parsing
parser = argparse.ArgumentParser(description="Experiment 3: Fine-tune Stanford Dogs model on Cats vs Dogs dataset")
parser.add_argument("log_dir", type=str, help="Directory to save logs")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
args = parser.parse_args()

# Dataset Path
dataset_path = "/datasets/PetImages"

# Load saved model from Experiment 1
saved_model_path = "Saved_model/stanford_dogs_model_final.keras"
stanford_model = tf.keras.models.load_model(saved_model_path)

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

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])


train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

inputs = stanford_model.input

# Replace the first two convolutional layers with new ones
x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(inputs)
x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)

x = layers.GlobalAveragePooling2D()(x)

output = layers.Dense(1, activation='sigmoid', name='output_layer')(x)

new_model = tf.keras.Model(inputs=inputs, outputs=output)

for layer in new_model.layers:
    if "conv2d" in layer.name and ("conv2d_0" in layer.name or "conv2d_1" in layer.name):
        layer.trainable = True
    else:
        layer.trainable = False

new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("fine_tuned_model.keras"),
    tf.keras.callbacks.TensorBoard(log_dir=args.log_dir),
]

new_model.fit(
    train_ds,
    epochs=args.epochs,
    validation_data=val_ds,
    callbacks=callbacks,
)

