from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from classic import warp_document
from neural_utils import draw_prediction, predict_contour

root = Path(__file__).parent
input_dir = root / "input"
benchmark_dir = root / "object_benchmark_516x600"
model_path = root / "model" / "contour_resnet50_synthetic_512x683.keras"
result_dir = root / "result"
result_dir.mkdir(exist_ok=True)

model = tf.keras.models.load_model(model_path, compile=False)
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
source_folders = [
    (input_dir, ""),
    (benchmark_dir, "object_benchmark_"),
]
processed_count = 0
error_count = 0

for source_dir, prefix in source_folders:
    if not source_dir.exists():
        continue

    if source_dir == benchmark_dir:
        found_images = source_dir.rglob("*")
    else:
        found_images = source_dir.iterdir()

    image_paths = []
    for path in found_images:
        if source_dir == benchmark_dir and path.parent == source_dir:
            continue
        if path.is_file() and path.suffix.lower() in image_extensions:
            image_paths.append(path)
    image_paths.sort()

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            error_count += 1
            continue

        try:
            corners, contour, visibility, prepared, model_area = predict_contour(
                image,
                model,
                use_color=False,
            )

            prepared_image = (prepared[:, :, 0] > 0).astype(np.uint8) * 255

            prediction = draw_prediction(
                image,
                contour,
                visibility,
                corners,
                model_area,
            )
            corrected = warp_document(image, corners)

            if source_dir == benchmark_dir:
                relative_path = image_path.relative_to(source_dir)
                parts = relative_path.with_suffix("").parts
                name = prefix + "_".join(parts)
            else:
                name = image_path.stem
            cv2.imwrite(
                str(result_dir / f"{name}_binary.jpg"),
                prepared_image,
            )
            cv2.imwrite(
                str(result_dir / f"{name}_points_and_corners.jpg"),
                prediction,
            )
            cv2.imwrite(
                str(result_dir / f"{name}_corrected.jpg"),
                corrected,
            )

            processed_count += 1

        except Exception as error:
            print("Ошибка при обработке", image_path.name, ":", error)
            error_count += 1

print("Обработано:", processed_count, "Ошибок:", error_count)
