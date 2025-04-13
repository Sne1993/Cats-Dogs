import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Experiment 1")
parser.add_argument("log_dir", type=str, help="Directory to save logs")
parser.add_argument("--epochs", type=int, default=25, help="Number of epochs for training")
args = parser.parse_args()

image_size = (180, 180)
batch_size = 64

# Load the Stanford Dogs dataset
train_ds, ds_info = tfds.load('stanford_dogs', split='train', as_supervised=True, with_info=True)
val_ds = tfds.load('stanford_dogs', split='test', as_supervised=True)

num_classes = ds_info.features['label'].num_classes


def preprocess(image, label):
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


train_ds = train_ds.map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

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

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=num_classes)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint("stanford_dogs_model.keras"),
    keras.callbacks.TensorBoard(log_dir=args.log_dir),
]

model.fit(
    train_ds,
    epochs=args.epochs,
    validation_data=val_ds,
    callbacks=callbacks,
)

# Save
model.save("stanford_dogs_model_final.keras")
