from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import lpips
import torch

from classic import detect_corners_hough, order_corners, warp_document
from neural_utils import (
    corners_to_contour,
    point_visibility,
    predict_contour,
)

root = Path(__file__).parent
benchmark_dir = root / "object_benchmark_516x600"
result_dir = root / "comparison_results"
result_dir.mkdir(exist_ok=True)

synthetic_model_path = (
    root / "model" / "contour_resnet50_synthetic_512x683.keras"
)

model_files = {
    "neural_synthetic": (
        synthetic_model_path,
        False,
    ),
    "neural_real_color": (
        root / "model" / "contour_resnet50_real_color.keras",
        True,
    ),
}

models = {}
for method, (model_path, use_color) in model_files.items():
    if model_path.exists():
        models[method] = {
            "model": tf.keras.models.load_model(model_path, compile=False),
            "use_color": use_color,
        }

lpips_model = lpips.LPIPS(net="alex").eval()

def predict_neural(image, method):
    settings = models[method]
    corners, contour, visibility, _, model_area = predict_contour(
        image,
        settings["model"],
        use_color=settings["use_color"],
    )
    return corners, contour, visibility, model_area

def corners_from_row(row):
    return np.float32([
        [row.tl_x, row.tl_y],
        [row.tr_x, row.tr_y],
        [row.br_x, row.br_y],
        [row.bl_x, row.bl_y],
    ])

def lpips_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor * 2 - 1

def sift_metrics(first, second):
    first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(first, None)
    keypoints2, descriptors2 = sift.detectAndCompute(second, None)

    if descriptors1 is None or descriptors2 is None:
        return 0, 0.0, np.nan

    matcher = cv2.FlannBasedMatcher(
        {"algorithm": 1, "trees": 5},
        {"checks": 50},
    )
    pairs = matcher.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for pair in pairs:
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            good_matches.append(pair[0])

    if not good_matches:
        return 0, 0.0, np.nan

    match_count = len(good_matches)

    match_ratio = match_count / len(keypoints1)
    average_distance = np.mean([match.distance for match in good_matches])
    return match_count, float(match_ratio), float(average_distance)

def compare_result(image, ideal_points, predicted_points):

    ideal = warp_document(image, ideal_points, output_size=(500, 700))
    predicted = warp_document(image, predicted_points, output_size=(500, 700))

    with torch.no_grad():
        lpips_value = lpips_model(
            lpips_tensor(ideal),
            lpips_tensor(predicted),
        ).item()

    sift_count, sift_ratio, sift_distance = sift_metrics(ideal, predicted)

    ideal_points = order_corners(ideal_points)
    predicted_points = order_corners(predicted_points)
    corner_error = np.linalg.norm(
        ideal_points - predicted_points,
        axis=1,
    ).mean()

    return {
        "lpips": float(lpips_value),
        "sift_matches": sift_count,
        "sift_ratio": sift_ratio,
        "sift_avg_distance": sift_distance,
        "corner_error_px": float(corner_error),
    }

annotations = pd.read_csv(benchmark_dir / "annotations.csv")
rows = []
methods = ["classic"] + list(models)

for row in annotations.itertuples(index=False):
    image_path = benchmark_dir / row.image
    image = cv2.imread(str(image_path))
    ideal_points = corners_from_row(row)

    for method in methods:
        result = {
            "image": row.image,
            "method": method,
            "condition": row.condition,
            "detected": False,
            "problem": "",
            "contour_error_px": np.nan,
            "visibility_accuracy": np.nan,
        }

        try:
            if method == "classic":
                predicted_points = detect_corners_hough(image)
            else:
                (
                    predicted_points,
                    predicted_contour,
                    predicted_visibility,
                    model_area,
                ) = predict_neural(image, method)

                ideal_contour = corners_to_contour(ideal_points)
                crop_x, crop_y, crop_width, crop_height = model_area
                contour_in_crop = ideal_contour.copy()
                contour_in_crop[:, 0] -= crop_x
                contour_in_crop[:, 1] -= crop_y
                ideal_visibility = point_visibility(
                    contour_in_crop,
                    crop_width,
                    crop_height,
                )
                result["contour_error_px"] = float(
                    np.linalg.norm(
                        ideal_contour - predicted_contour,
                        axis=1,
                    ).mean()
                )
                result["visibility_accuracy"] = float(
                    np.mean(
                        (predicted_visibility >= 0.5)
                        == ideal_visibility
                    )
                )

            result.update(
                compare_result(image, ideal_points, predicted_points)
            )
            result["detected"] = True

        except Exception as error:
            result["problem"] = type(error).__name__ + ": " + str(error)

        rows.append(result)

results = pd.DataFrame(rows)
results.to_csv(result_dir / "all_results.csv", index=False)

separate_files = {
    "classic": "classic_results.csv",
    "neural_synthetic": "synthetic_contour_model_results.csv",
    "neural_real_color": "real_color_contour_model_results.csv",
}
for method, filename in separate_files.items():
    method_results = results[results["method"] == method]
    if not method_results.empty:
        method_results.to_csv(result_dir / filename, index=False)

def get_summary(group_columns):
    summary = results.groupby(group_columns, dropna=False).agg(
        images=("image", "count"),
        detected=("detected", "sum"),
        lpips=("lpips", "mean"),
        sift_matches=("sift_matches", "mean"),
        sift_ratio=("sift_ratio", "mean"),
        sift_avg_distance=("sift_avg_distance", "mean"),
        corner_error_px=("corner_error_px", "mean"),
        contour_error_px=("contour_error_px", "mean"),
        visibility_accuracy=("visibility_accuracy", "mean"),
    ).reset_index()

    summary["detection_rate"] = summary["detected"] / summary["images"]
    return summary

by_condition = get_summary(["method", "condition"])
by_condition.to_csv(result_dir / "summary_by_condition.csv", index=False)

summary_files = {
    "classic": "classic_summary.csv",
    "neural_synthetic": "synthetic_contour_model_summary.csv",
    "neural_real_color": "real_color_contour_model_summary.csv",
}
for method, filename in summary_files.items():
    method_summary = by_condition[by_condition["method"] == method]
    if not method_summary.empty:
        method_summary.to_csv(result_dir / filename, index=False)

def save_plot():

    conditions = ["без объектов", "с объектами"]
    method_names = {
        "classic": ("Классический подход", "o"),
        "neural_synthetic": ("Контурная синтетическая модель", "x"),
        "neural_real_color": ("Контурная модель на реальных фото", "+"),
    }
    methods = [
        (method, *method_names[method])
        for method in results["method"].unique()
    ]
    colors = {
        "без объектов": "tab:green",
        "с объектами": "tab:red",
    }

    figure, axes = plt.subplots(
        len(methods),
        2,
        figsize=(12, 5 * len(methods)),
        squeeze=False,
    )

    for row_number, (method, method_name, marker) in enumerate(methods):
        for column_number, condition in enumerate(conditions):
            axis = axes[row_number, column_number]

            points = results[
                (results["method"] == method)
                & (results["condition"] == condition)
                & results["detected"]
            ]

            axis.scatter(
                points["lpips"],
                points["sift_ratio"],
                color=colors[condition],
                marker=marker,
                alpha=0.65,
            )

            axis.set_title("{}: {} (точек: {})".format(
                method_name,
                condition,
                len(points),
            ))
            axis.set_xlabel("LPIPS")
            axis.set_ylabel("SIFT match ratio")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.4)

    figure.suptitle("Сравнение способов определения контура листа")
    figure.tight_layout()
    figure.savefig(result_dir / "comparison_points.png", dpi=160)
    plt.close(figure)

save_plot()

print("\nРезультат по двум условиям:")
print(by_condition.to_string(index=False))
