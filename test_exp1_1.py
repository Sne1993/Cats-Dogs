import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import data as tf_data
from tensorflow import keras
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Experiment 1")
parser.add_argument("log_dir", type=str, help="Directory to save logs")
parser.add_argument("--epochs", type=int, default=25, help="Number of epochs for training")
args = parser.parse_args()

# Dataset Path
dataset_path = "datasets/PetImages"

def remove_corrupted_images(dataset_path):
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(dataset_path, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                # Try to open the image to check if it's corrupt
                img = tf.io.read_file(fpath)
                img = tf.image.decode_jpeg(img, channels=3)
            except tf.errors.InvalidArgumentError:
                num_skipped += 1
                os.remove(fpath)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                
    print(f"Deleted {num_skipped} corrupted images.")

remove_corrupted_images(dataset_path)

image_size = (180, 180)
batch_size = 64

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

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

inputs = keras.Input(shape=(image_size + (3,)))
x = data_augmentation(inputs)
x = layers.Rescaling(1.0 / 255)(x)

train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=2)

# Set learning rate to 0.0001
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[keras.metrics.BinaryAccuracy(name="acc")],
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint("experiment1_model_{epoch}.keras"),
]

model.fit(
    train_ds,
    epochs=args.epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

val_loss, val_acc = model.evaluate(val_ds)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_acc:.4f}')
