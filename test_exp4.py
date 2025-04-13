import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Argument parsing
parser = argparse.ArgumentParser(description="Experiment 4")
parser.add_argument("log_dir", type=str, help="Directory to save logs")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
args = parser.parse_args()

# Dataset path
dataset_path = "/datasets/PetImages"

# Load base model
saved_model_path = "Saved_model/stanford_dogs_model_final.keras"
model = tf.keras.models.load_model(saved_model_path)

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

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

c = 2

x = model.input
for layer in model.layers[::-1]:
    if c != 0 and ("conv2d" in layer.name or "separable_conv2d" in layer.name):
        if "separable_conv2d" in layer.name:
            new_layer = layers.SeparableConv2D(1024, 3, padding="same", activation='relu')
        else:
            new_layer = layers.Conv2D(728, 1, strides=2, padding="same", activation='relu')

        new_layer.build(layer.input.shape)

        if new_layer.get_weights() and layer.get_weights():
            new_layer.set_weights(layer.get_weights())

        layer.trainable = True
        c -= 1
    else:
        if layer.name != 'dense':
            layer.trainable = False

output = tf.keras.layers.Dense(1, activation='sigmoid')(model.layers[-2].output)
output.trainable = True

new_model = models.Model(inputs=model.input, outputs=output)


new_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="acc")],
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint("fine_tuned_model.keras"),
    keras.callbacks.TensorBoard(log_dir=args.log_dir),
]

new_model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=callbacks)

# Save
new_model.save("fine_tuned_model_final.keras")
