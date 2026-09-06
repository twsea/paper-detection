import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 224
CANVAS_WIDTH = 512
CANVAS_HEIGHT = 683
POINTS_PER_SIDE = 50
CONTOUR_POINT_COUNT = POINTS_PER_SIDE * 4

def prepare_binary_image(image):

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    scale = min(CANVAS_WIDTH / width, CANVAS_HEIGHT / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = cv2.resize(image, (new_width, new_height))

    image = cv2.GaussianBlur(image, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        5,
    )

    offset_x = (CANVAS_WIDTH - new_width) // 2
    offset_y = (CANVAS_HEIGHT - new_height) // 2

    canvas = np.full(
        (CANVAS_HEIGHT, CANVAS_WIDTH),
        255,
        dtype=np.uint8,
    )
    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = binary

    canvas = cv2.resize(
        canvas,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    canvas = canvas.astype(np.float32)
    prepared = tf.keras.applications.resnet50.preprocess_input(canvas)

    transform = (
        new_width / width,
        new_height / height,
        offset_x,
        offset_y,
    )
    return prepared, transform

def prepare_color_image(image):

    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    return tf.keras.applications.resnet50.preprocess_input(image)

def corners_to_contour(corners):

    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    positions = np.linspace(
        0.0,
        1.0,
        POINTS_PER_SIDE,
        endpoint=False,
        dtype=np.float32,
    )

    sides = []
    for side_number in range(4):
        start = corners[side_number]
        end = corners[(side_number + 1) % 4]
        side = start + positions[:, None] * (end - start)
        sides.append(side)

    return np.concatenate(sides).astype(np.float32)

def point_visibility(points, width, height):

    points = np.asarray(points)
    return (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    ).astype(np.float32)

def build_contour_model():
    inputs = tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    resnet = tf.keras.applications.ResNet50(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )

    resnet.trainable = False

    x = resnet(inputs, training=False)
    x = tf.keras.layers.Conv2D(32, 1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    point_values = tf.keras.layers.Dense(
        CONTOUR_POINT_COUNT * 2,
    )(x)
    points = tf.keras.layers.Reshape(
        (CONTOUR_POINT_COUNT, 2),
        name="points",
    )(point_values)
    visibility = tf.keras.layers.Dense(
        CONTOUR_POINT_COUNT,
        activation="sigmoid",
        name="visibility",
    )(x)

    return tf.keras.Model(
        inputs,
        {"points": points, "visibility": visibility},
    )

def fit_side_line(points):

    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32),
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)

    return float(vy), float(-vx), float(vx * y0 - vy * x0)

def intersect_lines(first, second, fallback):

    a1, b1, c1 = first
    a2, b2, c2 = second
    determinant = a1 * b2 - a2 * b1

    if abs(determinant) < 1e-6:
        return np.asarray(fallback, dtype=np.float32)

    x = (b1 * c2 - b2 * c1) / determinant
    y = (c1 * a2 - c2 * a1) / determinant
    return np.float32([x, y])

def contour_to_corners(points):

    points = np.asarray(points, dtype=np.float32).reshape(
        CONTOUR_POINT_COUNT,
        2,
    )

    side_points = []
    for side_number in range(4):
        start = side_number * POINTS_PER_SIDE
        end = start + POINTS_PER_SIDE
        side_points.append(points[start:end])

    lines = [
        fit_side_line(side_points[number])
        for number in range(4)
    ]

    return np.float32([
        intersect_lines(lines[0], lines[3], points[0]),
        intersect_lines(lines[0], lines[1], points[POINTS_PER_SIDE]),
        intersect_lines(lines[2], lines[1], points[POINTS_PER_SIDE * 2]),
        intersect_lines(lines[2], lines[3], points[POINTS_PER_SIDE * 3]),
    ])

def order_corners(points):

    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(
        points[:, 1] - center[1],
        points[:, 0] - center[0],
    )
    ordered = points[np.argsort(angles)]
    top_left_index = np.argmin(ordered[:, 0] + ordered[:, 1])
    return np.roll(ordered, -top_left_index, axis=0).astype(np.float32)

def predict_contour(image, model, use_color=False):
    if image is None:
        raise ValueError("Изображение не загрузилось")

    if use_color:
        prepared = prepare_color_image(image)
    else:
        prepared, transform = prepare_binary_image(image)

    prediction = model.predict(prepared[None, ...], verbose=0)
    if not isinstance(prediction, dict):
        prediction = dict(zip(model.output_names, prediction))

    points = np.float32(prediction["points"][0])
    visibility = np.float32(prediction["visibility"][0])
    height, width = image.shape[:2]

    if use_color:
        points[:, 0] *= width
        points[:, 1] *= height
    else:

        points[:, 0] *= CANVAS_WIDTH
        points[:, 1] *= CANVAS_HEIGHT

        scale_x, scale_y, offset_x, offset_y = transform
        points[:, 0] = (points[:, 0] - offset_x) / scale_x
        points[:, 1] = (points[:, 1] - offset_y) / scale_y

    model_area = (0, 0, width, height)
    corners = order_corners(contour_to_corners(points))
    return corners, points, visibility, prepared, model_area

def draw_prediction(
    image,
    points,
    visibility,
    corners,
    model_area,
):

    height, width = image.shape[:2]
    all_coordinates = np.vstack((
        points,
        corners,
        np.float32([[0, 0], [width, height]]),
    ))

    margin = 35
    minimum = np.floor(all_coordinates.min(axis=0) - margin)
    maximum = np.ceil(all_coordinates.max(axis=0) + margin)
    drawing_size = maximum - minimum

    scale = min(
        1.0,
        1600 / drawing_size[0],
        1200 / drawing_size[1],
    )
    canvas_width = max(1, int(np.ceil(drawing_size[0] * scale)))
    canvas_height = max(1, int(np.ceil(drawing_size[1] * scale)))
    result = np.full(
        (canvas_height, canvas_width, 3),
        35,
        dtype=np.uint8,
    )

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_image = cv2.resize(image, (resized_width, resized_height))
    image_start = np.rint(-minimum * scale).astype(np.int32)
    image_x, image_y = image_start
    result[
        image_y:image_y + resized_height,
        image_x:image_x + resized_width,
    ] = resized_image

    crop_x, crop_y, crop_width, crop_height = model_area
    frame_start = np.rint(
        (np.float32([crop_x, crop_y]) - minimum) * scale
    ).astype(np.int32)
    frame_end = np.rint(
        (
            np.float32([crop_x + crop_width, crop_y + crop_height])
            - minimum
        ) * scale
    ).astype(np.int32)

    cv2.rectangle(
        result,
        tuple(frame_start),
        tuple(frame_end),
        (255, 220, 0),
        2,
    )

    legend = [
        ((0, 255, 0), "in frame"),
        ((255, 80, 0), "out of frame"),
        ((0, 0, 255), "corner"),
    ]
    legend_x = 12
    for color, text in legend:
        cv2.circle(result, (legend_x, 17), 4, color, -1)
        cv2.putText(
            result,
            text,
            (legend_x + 8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        legend_x += 125

    rounded = np.rint((points - minimum) * scale).astype(np.int32)

    for number, point in enumerate(rounded):
        color = (0, 255, 0) if visibility[number] >= 0.5 else (255, 80, 0)
        cv2.circle(result, tuple(point), 3, color, -1)

    corner_names = ["TL", "TR", "BR", "BL"]
    drawn_corners = np.rint((corners - minimum) * scale).astype(np.int32)
    for name, point in zip(
        corner_names,
        drawn_corners,
    ):
        cv2.circle(result, tuple(point), 7, (0, 0, 255), -1)
        cv2.putText(
            result,
            name,
            (point[0] + 9, point[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return result
