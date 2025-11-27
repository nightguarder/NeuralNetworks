# Test TensorFlow with Metal (Apple Silicon) - reduced for faster runtime
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tensorflow as tf

cifar = keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

# Use a smaller subset for quick tests
x_train = x_train[:2048]
y_train = y_train[:2048]
x_test = x_test[:512]
y_test = y_test[:512]

model = keras.applications.ResNet50(
    include_top=True,
    weights=None,  # type: ignore
    input_shape=(32, 32, 3),
    classes=100,
)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Fewer epochs and larger batch for shorter test runs
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_data=(x_test, y_test))
## The output should show training progress without errors.