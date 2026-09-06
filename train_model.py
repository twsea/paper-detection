from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv

import cv2
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from neural_utils import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    CONTOUR_POINT_COUNT,
    IMAGE_SIZE,
    build_contour_model,
    corners_to_contour,
    point_visibility,
    prepare_binary_image,
)

seed = 500
np.random.seed(seed)
tf.random.set_seed(seed)

project_dir = Path(__file__).parent
dataset_dir = project_dir / "dataset_generated"
model_dir = project_dir / "model"
model_dir.mkdir(exist_ok=True)

batch_size = 64
pool = ThreadPoolExecutor(max_workers=6)
corners = np.load(dataset_dir / "corners.npz")["corners"]

def read_item(index):

    image_path = dataset_dir / f"img_{index:06d}.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    points = corners_to_contour(corners[index])
    visibility = point_visibility(points, width, height)

    prepared, transform = prepare_binary_image(image)
    scale_x, scale_y, offset_x, offset_y = transform
    points[:, 0] = points[:, 0] * scale_x + offset_x
    points[:, 1] = points[:, 1] * scale_y + offset_y
    points[:, 0] /= CANVAS_WIDTH
    points[:, 1] /= CANVAS_HEIGHT

    targets = {
        "points": points,
        "visibility": visibility,
    }
    return prepared, targets

def batches(indices, shuffle=True):

    indices = indices.copy()

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            examples = list(pool.map(read_item, batch_indices))
            images, targets = zip(*examples)

            yield (
                np.asarray(images, dtype=np.float32),
                {
                    "points": np.asarray(
                        [target["points"] for target in targets],
                        dtype=np.float32,
                    ),
                    "visibility": np.asarray(
                        [target["visibility"] for target in targets],
                        dtype=np.float32,
                    ),
                },
            )

def get_dataset(indices, shuffle=True):

    dataset = tf.data.Dataset.from_generator(
        lambda: batches(indices, shuffle),
        output_signature=(
            tf.TensorSpec(
                shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3),
                dtype=tf.float32,
            ),
            {
                "points": tf.TensorSpec(
                    shape=(None, CONTOUR_POINT_COUNT, 2),
                    dtype=tf.float32,
                ),
                "visibility": tf.TensorSpec(
                    shape=(None, CONTOUR_POINT_COUNT),
                    dtype=tf.float32,
                ),
            },
        ),
    )
    return dataset.prefetch(1)

image_count = 100000
indices = np.arange(image_count)

train_indices, other_indices = train_test_split(
    indices,
    test_size=0.10,
    random_state=seed,
)
validation_indices, test_indices = train_test_split(
    other_indices,
    test_size=0.50,
    random_state=seed,
)

print("Train/validation/test:", len(train_indices), len(validation_indices), len(test_indices))

train_steps = int(np.ceil(len(train_indices) / batch_size))
validation_steps = int(np.ceil(len(validation_indices) / batch_size))
test_steps = int(np.ceil(len(test_indices) / batch_size))

train_data = get_dataset(train_indices)
validation_data = get_dataset(validation_indices, shuffle=False)
test_data = get_dataset(test_indices, shuffle=False)

model = build_contour_model()
best_weights_path = (
    model_dir / "contour_resnet50_synthetic_512x683_best.weights.h5"
)
best_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_weights_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        "points": "mean_squared_error",
        "visibility": "binary_crossentropy",
    },
    loss_weights={"points": 1.0, "visibility": 0.05},
    metrics={"points": ["mae"], "visibility": ["accuracy"]},
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

model.fit(
    train_data,
    steps_per_epoch=train_steps,
    validation_data=validation_data,
    validation_steps=validation_steps,
    epochs=10,
    callbacks=[early_stopping, best_checkpoint],
)

resnet = model.get_layer("resnet50")
resnet.trainable = True

for layer in resnet.layers:
    layer.trainable = (
        layer.name.startswith("conv5_")
        and not isinstance(layer, tf.keras.layers.BatchNormalization)
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss={
        "points": "mean_squared_error",
        "visibility": "binary_crossentropy",
    },
    loss_weights={"points": 1.0, "visibility": 0.05},
    metrics={"points": ["mae"], "visibility": ["accuracy"]},
)

fine_tune_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

model.fit(
    train_data,
    steps_per_epoch=train_steps,
    validation_data=validation_data,
    validation_steps=validation_steps,
    epochs=5,
    callbacks=[fine_tune_stopping, best_checkpoint],
)

model.load_weights(best_weights_path)

result = model.evaluate(
    test_data,
    steps=test_steps,
    verbose=0,
    return_dict=True,
)
model.save(model_dir / "contour_resnet50_synthetic_512x683.keras")

with (model_dir / "synthetic_contour_test_metrics.csv").open(
    "w",
    encoding="utf-8",
    newline="",
) as file:
    writer = csv.writer(file)
    writer.writerow(["metric", "value"])
    for name, value in result.items():
        writer.writerow([name, value])

print("Результат на test:", result)
print("Модель и метрики сохранены в папку model")
