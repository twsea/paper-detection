from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import random

import cv2
import numpy as np

generator_settings = {
    "output_dir": "dataset_generated",
    "image_count": 100000,

    "image_width": 683,
    "image_height": 683,
    "paper_width": 420,
    "paper_height": 594,
    "minimum_visible_part": 0.50,
    "full_large_maximum": 0.99,
    "partial_large_maximum": 1.10,

    "usual_perspective_change": 0.12,
    "strong_perspective_change": 0.18,
    "extreme_perspective_change": 0.25,
    "maximum_opposite_side_ratio": 1.65,
    "blank_paper_probability": 0.5,
    "seed": 500,
    "worker_count": 8,
}

def random_image_size(settings, image_format):

    maximum_width = settings["image_width"]
    maximum_height = settings["image_height"]

    if image_format == "landscape":

        width = random.randint(400, maximum_width)
        minimum_height = max(220, int(width * 0.45))
        maximum_landscape_height = min(
            maximum_height,
            int(width * 0.88),
        )
        height = random.randint(minimum_height, maximum_landscape_height)

    elif image_format == "square":

        base_size = random.randint(330, min(maximum_width, maximum_height))
        width = int(base_size * random.uniform(0.90, 1.10))
        height = int(base_size * random.uniform(0.90, 1.10))
        width = min(width, maximum_width)
        height = min(height, maximum_height)

    else:

        height = random.randint(450, maximum_height)
        minimum_width = max(260, int(height * 0.48))
        maximum_portrait_width = min(maximum_width, int(height * 0.90))
        width = random.randint(minimum_width, maximum_portrait_width)

    return width, height

def random_paper_size(settings):

    width = settings["paper_width"]

    if random.random() < 0.50:
        ratio = settings["paper_height"] / width
    else:
        ratio = random.uniform(1.20, 1.60)

    height = int(round(width * ratio))
    return width, height

def random_paper_angle(orientation):

    if orientation == "horizontal":
        return random.choice([-90, 90]) + random.uniform(-20, 20)

    return random.choice([0, 180]) + random.uniform(-55, 55)

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

def visible_fraction(quad, width, height):

    full_area = abs(cv2.contourArea(quad))
    if full_area == 0:
        return 0.0

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    return cv2.countNonZero(mask) / full_area

def change_perspective(quad, settings, perspective_group):

    if perspective_group == "usual":
        minimum = 0.0
        maximum = settings["usual_perspective_change"]
    elif perspective_group == "strong":
        minimum = settings["usual_perspective_change"]
        maximum = settings["strong_perspective_change"]
    else:
        minimum = settings["strong_perspective_change"]
        maximum = settings["extreme_perspective_change"]

    perspective = random.uniform(minimum, maximum)
    direction_angle = random.uniform(0, 2 * np.pi)
    direction = np.float32([
        np.cos(direction_angle),
        np.sin(direction_angle),
    ])
    normalizer = abs(direction[0]) + abs(direction[1])
    unit_corners = np.float32([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ])

    depth = 1 + perspective * (unit_corners @ direction) / normalizer
    return quad / depth[:, None]

def fallback_quad(
    settings,
    outside_count,
    size_group,
    orientation,
    perspective_group,
):

    width = settings["image_width"]
    height = settings["image_height"]
    paper_ratio = settings["paper_height"] / settings["paper_width"]
    area_part = {
        "small": 0.14,
        "medium": 0.32,
        "large": 0.62,
    }[size_group]

    paper_width = np.sqrt(area_part * width * height / paper_ratio)
    paper_height = paper_width * paper_ratio
    quad = np.float32([
        [-paper_width / 2, -paper_height / 2],
        [paper_width / 2, -paper_height / 2],
        [paper_width / 2, paper_height / 2],
        [-paper_width / 2, paper_height / 2],
    ])
    quad = change_perspective(quad, settings, perspective_group)

    if orientation == "horizontal":
        angle = random.choice([-75, 75])
    else:
        angle = random.choice([-15, 15])

    angle = np.deg2rad(angle)
    rotation = np.float32([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    quad = quad @ rotation.T

    span_x, span_y = np.ptp(quad, axis=0)
    maximum_part = 0.86 if outside_count == 0 else 0.75
    scale = min(
        1.0,
        width * maximum_part / span_x,
        height * maximum_part / span_y,
    )
    quad *= scale

    min_x, min_y = quad.min(axis=0)
    max_x, max_y = quad.max(axis=0)
    quad[:, 0] += width / 2 - (min_x + max_x) / 2

    if outside_count == 0:
        quad[:, 1] += height / 2 - (min_y + max_y) / 2
    else:

        sorted_y = np.sort(quad[:, 1])
        first_inside = outside_count
        gap = sorted_y[first_inside] - sorted_y[first_inside - 1]
        shift_y = -sorted_y[first_inside - 1] - gap * 0.08
        quad[:, 1] += shift_y

    return quad

def random_paper_position(
    settings,
    outside_count,
    size_group,
    orientation,
    perspective_group,
):

    width = settings["image_width"]
    height = settings["image_height"]
    paper_ratio = settings["paper_height"] / settings["paper_width"]

    if outside_count == 0:
        large_maximum = settings["full_large_maximum"]
    else:
        large_maximum = settings["partial_large_maximum"]
    size_ranges = {
        "small": (0.08, 0.20),
        "medium": (0.20, 0.45),
        "large": (0.45, large_maximum),
    }
    minimum_area, maximum_area = size_ranges[size_group]

    for _ in range(600):
        area_part = random.uniform(minimum_area, maximum_area)

        quad_width = np.sqrt(area_part * width * height / paper_ratio)
        quad_height = quad_width * paper_ratio

        quad = np.float32([
            [-quad_width / 2, -quad_height / 2],
            [quad_width / 2, -quad_height / 2],
            [quad_width / 2, quad_height / 2],
            [-quad_width / 2, quad_height / 2],
        ])

        quad = change_perspective(quad, settings, perspective_group)

        angle = np.deg2rad(random_paper_angle(orientation))
        rotation = np.float32([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        quad = quad @ rotation.T

        current_area = abs(cv2.contourArea(quad))
        if current_area == 0:
            continue
        scale = np.sqrt(area_part * width * height / current_area)
        quad *= scale

        min_x, min_y = quad.min(axis=0)
        max_x, max_y = quad.max(axis=0)

        if outside_count == 0:

            if max_x - min_x >= width or max_y - min_y >= height:
                continue

            center_x = random.uniform(-min_x, width - max_x)
            center_y = random.uniform(-min_y, height - max_y)
            quad += np.float32([center_x, center_y])

        else:

            side = random.choice(["left", "right", "top", "bottom"])

            if side in ("left", "right"):
                if max_y - min_y >= height:
                    continue

                center_y = random.uniform(-min_y, height - max_y)
                coordinates = np.sort(quad[:, 0])

                if side == "left":
                    first_center = -coordinates[outside_count]
                    second_center = -coordinates[outside_count - 1]
                else:
                    first_outside = 4 - outside_count
                    first_center = width - coordinates[first_outside]
                    second_center = width - coordinates[first_outside - 1]

                center_x = random.uniform(first_center, second_center)

            else:
                if max_x - min_x >= width:
                    continue

                center_x = random.uniform(-min_x, width - max_x)
                coordinates = np.sort(quad[:, 1])

                if side == "top":
                    first_center = -coordinates[outside_count]
                    second_center = -coordinates[outside_count - 1]
                else:
                    first_outside = 4 - outside_count
                    first_center = height - coordinates[first_outside]
                    second_center = height - coordinates[first_outside - 1]

                center_y = random.uniform(first_center, second_center)

            quad += np.float32([center_x, center_y])

        inside = (
            (quad[:, 0] >= 0)
            & (quad[:, 0] < width)
            & (quad[:, 1] >= 0)
            & (quad[:, 1] < height)
        )
        actual_outside_count = int(4 - inside.sum())

        sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        opposite_side_ratio = max(
            max(sides[0], sides[2]) / min(sides[0], sides[2]),
            max(sides[1], sides[3]) / min(sides[1], sides[3]),
        )
        enough_visible = (
            visible_fraction(quad, width, height)
            >= settings["minimum_visible_part"]
        )

        if (
            cv2.isContourConvex(quad.astype(np.int32))
            and sides.min() > 50
            and opposite_side_ratio <= settings["maximum_opposite_side_ratio"]
            and enough_visible
            and actual_outside_count == outside_count
        ):
            return quad, False

    return (
        fallback_quad(
            settings,
            outside_count,
            size_group,
            orientation,
            perspective_group,
        ),
        True,
    )

def create_background(settings):

    height = settings["image_height"]
    width = settings["image_width"]

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = x / width - 0.5
    y = y / height - 0.5

    base_color = random.randint(25, 240)
    background = np.full((height, width), base_color, dtype=np.float32)

    background += x * random.uniform(-90, 90)
    background += y * random.uniform(-90, 90)

    rough_noise = np.random.normal(0, 1, (4, 4)).astype(np.float32)
    rough_noise = cv2.resize(rough_noise, (width, height))
    rough_noise = cv2.GaussianBlur(rough_noise, (0, 0), 50)
    background += rough_noise * random.uniform(10, 38)

    background += np.random.normal(
        0,
        random.uniform(0.5, 3.0),
        background.shape,
    )

    background = np.clip(background, 0, 255).astype(np.uint8)

    surface = random.choice(["smooth", "smooth", "wood", "tiles", "fabric"])

    if surface == "wood":
        line_color = int(np.median(background)) + random.randint(-35, 35)
        line_color = int(np.clip(line_color, 0, 255))

        for _ in range(random.randint(4, 10)):
            y1 = random.randrange(height)
            y2 = int(np.clip(y1 + random.randint(-35, 35), 0, height - 1))
            cv2.line(
                background,
                (0, y1),
                (width - 1, y2),
                line_color,
                random.randint(1, 5),
            )

    elif surface == "tiles":
        seam_color = int(np.clip(base_color + random.randint(-55, -20), 0, 255))
        x1 = random.randint(width // 4, 3 * width // 4)
        y1 = random.randint(height // 4, 3 * height // 4)
        cv2.line(background, (x1, 0), (x1, height - 1), seam_color, random.randint(3, 8))
        cv2.line(background, (0, y1), (width - 1, y1), seam_color, random.randint(3, 8))

    elif surface == "fabric":
        line_color = int(np.clip(base_color + random.randint(-30, 30), 0, 255))
        distance = random.randint(25, 55)
        for start_x in range(-height, width, distance):
            cv2.line(
                background,
                (start_x, 0),
                (start_x + height, height - 1),
                line_color,
                random.randint(1, 3),
            )

    if random.random() < 0.55:
        shadow = np.zeros_like(background)
        center = (random.randrange(width), random.randrange(height))
        axes = (
            random.randint(width // 4, width),
            random.randint(height // 5, height // 2),
        )
        cv2.ellipse(shadow, center, axes, random.randint(0, 180), 0, 360, 255, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), random.randint(25, 60))
        darkness = random.randint(15, 65)
        background = np.clip(
            background.astype(np.float32) - shadow / 255 * darkness,
            0,
            255,
        ).astype(np.uint8)

    if random.random() < 0.55:
        glare = np.zeros_like(background)
        center = (random.randrange(width), random.randrange(height))
        axes = (
            random.randint(width // 4, width // 2),
            random.randint(height // 10, height // 3),
        )

        cv2.ellipse(
            glare,
            center,
            axes,
            random.randint(0, 180),
            0,
            360,
            random.randint(35, 100),
            -1,
        )
        glare = cv2.GaussianBlur(glare, (0, 0), random.randint(20, 50))
        background = cv2.add(background, glare)

    return background

def create_paper(settings):

    height = settings["paper_height"]
    width = settings["paper_width"]

    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x = x / width - 0.5
    y = y / height - 0.5

    paper = np.full(
        (height, width),
        random.randint(205, 252),
        dtype=np.float32,
    )

    paper += x * random.uniform(-20, 20)
    paper += y * random.uniform(-20, 20)
    paper += np.random.normal(0, random.uniform(1, 4), paper.shape)
    paper = np.clip(paper, 0, 255).astype(np.uint8)

    if random.random() < settings["blank_paper_probability"]:
        return paper

    ink = random.randint(15, 100)
    left = random.randint(25, 50)
    right = width - random.randint(25, 50)

    if random.random() < 0.85:
        header_y = random.randint(25, 45)
        cv2.line(
            paper,
            (left, header_y),
            (random.randint(width // 2, right), header_y),
            ink,
            random.randint(3, 6),
        )

    current_y = random.randint(65, 90)
    while current_y < height - 35:
        line_count = random.randint(3, 9)

        for _ in range(line_count):
            if current_y >= height - 30:
                break

            line_end = random.randint(width // 2, right)
            line_color = int(np.clip(ink + random.randint(-10, 20), 0, 255))
            cv2.line(
                paper,
                (left + random.randint(0, 8), current_y),
                (line_end, current_y),
                line_color,
                random.randint(1, 2),
            )
            current_y += random.randint(9, 15)

        current_y += random.randint(12, 28)

    if random.random() < 0.45:
        box_width = random.randint(80, 170)
        box_height = random.randint(50, 130)
        x1 = random.randint(left, max(left, right - box_width))
        y1 = random.randint(80, max(80, height - box_height - 30))
        cv2.rectangle(
            paper,
            (x1, y1),
            (x1 + box_width, y1 + box_height),
            ink,
            random.randint(1, 3),
        )

    if random.random() < 0.35:
        for _ in range(random.randint(1, 3)):
            block_width = random.randint(55, 150)
            block_height = random.randint(45, 120)
            x1 = random.randint(left, max(left, right - block_width))
            y1 = random.randint(55, max(55, height - block_height - 25))
            color = random.randint(20, 125)
            cv2.rectangle(
                paper,
                (x1, y1),
                (x1 + block_width, y1 + block_height),
                color,
                -1,
            )

            for _ in range(random.randint(1, 4)):
                center = (
                    random.randint(x1, x1 + block_width),
                    random.randint(y1, y1 + block_height),
                )
                radius = random.randint(4, max(5, min(block_width, block_height) // 5))
                cv2.circle(paper, center, radius, random.randint(140, 235), -1)

    return paper

def rectangle_corners(center, width, height, angle):

    points = np.float32([
        [-width / 2, -height / 2],
        [width / 2, -height / 2],
        [width / 2, height / 2],
        [-width / 2, height / 2],
    ])

    angle = np.deg2rad(angle)
    rotation = np.float32([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    return points @ rotation.T + np.float32(center)

def draw_inside_lines(image, corners, color, line_count=5):

    for number in range(1, line_count + 1):
        part = number / (line_count + 1)
        left = corners[0] + part * (corners[3] - corners[0])
        right = corners[1] + part * (corners[2] - corners[1])
        cv2.line(
            image,
            tuple(left.astype(np.int32)),
            tuple(right.astype(np.int32)),
            color,
            2,
        )

def add_large_object(image, paper_corners):

    image_height, image_width = image.shape[:2]

    top_width = np.linalg.norm(paper_corners[1] - paper_corners[0])
    bottom_width = np.linalg.norm(paper_corners[2] - paper_corners[3])
    left_height = np.linalg.norm(paper_corners[3] - paper_corners[0])
    right_height = np.linalg.norm(paper_corners[2] - paper_corners[1])
    paper_width = (top_width + bottom_width) / 2
    paper_height = (left_height + right_height) / 2

    object_type = random.choice([
        "phone",
        "book",
        "second_paper",
        "package",
        "large_object",
    ])

    if object_type == "phone":
        object_width = paper_width * random.uniform(0.25, 0.45)
        object_height = paper_height * random.uniform(0.55, 0.90)
    elif object_type == "book":
        object_width = paper_width * random.uniform(0.55, 0.95)
        object_height = paper_height * random.uniform(0.55, 0.95)
    elif object_type == "second_paper":
        object_width = paper_width * random.uniform(0.65, 1.00)
        object_height = paper_height * random.uniform(0.65, 1.00)
    elif object_type == "package":
        object_width = paper_width * random.uniform(0.50, 0.90)
        object_height = paper_height * random.uniform(0.30, 0.70)
    else:

        object_width = paper_width * random.uniform(1.05, 1.30)
        object_height = paper_height * random.uniform(1.05, 1.30)

    object_width = float(np.clip(object_width, 45, image_width * 1.25))
    object_height = float(np.clip(object_height, 55, image_height * 1.25))
    center = (
        random.randint(0, image_width - 1),
        random.randint(0, image_height - 1),
    )
    corners = rectangle_corners(
        center,
        object_width,
        object_height,
        random.uniform(-60, 60),
    )

    shadow = (corners + np.float32([8, 10])).astype(np.int32)
    cv2.fillConvexPoly(image, shadow, random.randint(5, 45))

    if object_type == "phone":
        cv2.fillConvexPoly(image, corners.astype(np.int32), random.randint(5, 35))
        center_point = corners.mean(axis=0)
        screen = center_point + (corners - center_point) * 0.82
        cv2.fillConvexPoly(image, screen.astype(np.int32), random.randint(45, 130))
        cv2.polylines(image, [corners.astype(np.int32)], True, 210, 3)

    elif object_type == "second_paper":
        cv2.fillConvexPoly(image, corners.astype(np.int32), random.randint(215, 250))
        cv2.polylines(image, [corners.astype(np.int32)], True, random.randint(55, 120), 3)
        draw_inside_lines(image, corners, random.randint(20, 90), random.randint(6, 12))

    elif object_type == "book":
        color = random.randint(45, 190)
        outline = 235 if color < 120 else 30
        cv2.fillConvexPoly(image, corners.astype(np.int32), color)
        cv2.polylines(image, [corners.astype(np.int32)], True, outline, 4)
        cv2.line(
            image,
            tuple(corners[0].astype(np.int32)),
            tuple(corners[3].astype(np.int32)),
            outline,
            6,
        )
        draw_inside_lines(image, corners, outline, random.randint(2, 5))

    else:
        color = random.randint(55, 215)
        outline = 240 if color < 130 else 25
        cv2.fillConvexPoly(image, corners.astype(np.int32), color)
        cv2.polylines(image, [corners.astype(np.int32)], True, outline, 4)
        draw_inside_lines(image, corners, outline, random.randint(3, 7))

        center_point = corners.mean(axis=0)
        label = center_point + (corners - center_point) * 0.42
        cv2.fillConvexPoly(image, label.astype(np.int32), outline)

    return image

def add_small_objects(image, minimum=1, maximum=4):

    height, width = image.shape[:2]

    for _ in range(random.randint(minimum, maximum)):
        shape = random.choice(["circle", "ellipse", "rectangle", "line"])

        if random.random() < 0.65:
            color = random.randint(5, 100)
        else:
            color = random.randint(155, 245)

        shadow_color = random.randint(0, 55)
        outline_color = color + 65 if color < 128 else color - 65
        outline_color = int(np.clip(outline_color, 0, 255))

        if shape == "circle":
            center = (random.randrange(width), random.randrange(height))
            radius = random.randint(12, 55)
            cv2.circle(image, (center[0] + 6, center[1] + 7), radius, shadow_color, -1)
            cv2.circle(image, center, radius, color, -1)
            cv2.circle(image, center, radius, outline_color, random.randint(1, 3))
            cv2.circle(
                image,
                (center[0] - radius // 3, center[1] - radius // 3),
                max(2, radius // 7),
                min(255, color + 45),
                -1,
            )

        elif shape == "ellipse":
            center = (random.randrange(width), random.randrange(height))
            axes = (random.randint(20, 80), random.randint(10, 50))
            angle = random.randint(0, 180)
            cv2.ellipse(
                image,
                (center[0] + 6, center[1] + 7),
                axes,
                angle,
                0,
                360,
                shadow_color,
                -1,
            )
            cv2.ellipse(
                image,
                center,
                axes,
                angle,
                0,
                360,
                color,
                -1,
            )
            cv2.ellipse(image, center, axes, angle, 0, 360, outline_color, 2)

        elif shape == "rectangle":
            x1 = random.randrange(width)
            y1 = random.randrange(height)
            x2 = min(width - 1, x1 + random.randint(25, 130))
            y2 = min(height - 1, y1 + random.randint(20, 100))
            cv2.rectangle(
                image,
                (min(width - 1, x1 + 7), min(height - 1, y1 + 8)),
                (min(width - 1, x2 + 7), min(height - 1, y2 + 8)),
                shadow_color,
                -1,
            )
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(image, (x1, y1), (x2, y2), outline_color, 2)

            for line_y in range(y1 + 10, y2 - 5, random.randint(9, 16)):
                cv2.line(
                    image,
                    (x1 + 7, line_y),
                    (max(x1 + 8, x2 - 7), line_y),
                    outline_color,
                    1,
                )

        else:
            start = (random.randrange(width), random.randrange(height))
            end = (random.randrange(width), random.randrange(height))
            thickness = random.randint(5, 18)
            cv2.line(
                image,
                (start[0] + 5, start[1] + 6),
                (end[0] + 5, end[1] + 6),
                shadow_color,
                thickness + 3,
            )
            cv2.line(image, start, end, color, thickness)

    return image

def add_cable(image, paper_corners):

    height, width = image.shape
    paper_center = paper_corners.mean(axis=0)
    point_count = 8

    if random.random() < 0.65:
        x = np.linspace(-30, width + 30, point_count)
        base_y = paper_center[1] + random.uniform(-height * 0.25, height * 0.25)
        amplitude = random.uniform(height * 0.04, height * 0.16)
        phase = random.uniform(0, np.pi * 2)
        y = base_y + np.sin(np.linspace(0, np.pi * 2, point_count) + phase) * amplitude
    else:
        y = np.linspace(-30, height + 30, point_count)
        base_x = paper_center[0] + random.uniform(-width * 0.25, width * 0.25)
        amplitude = random.uniform(width * 0.04, width * 0.16)
        phase = random.uniform(0, np.pi * 2)
        x = base_x + np.sin(np.linspace(0, np.pi * 2, point_count) + phase) * amplitude

    points = np.column_stack((x, y)).astype(np.int32)
    thickness = random.randint(5, 14)

    cv2.polylines(
        image,
        [points + np.int32([4, 5])],
        False,
        random.randint(0, 40),
        thickness + 4,
        cv2.LINE_AA,
    )
    cv2.polylines(
        image,
        [points],
        False,
        random.randint(5, 80),
        thickness,
        cv2.LINE_AA,
    )
    return image

def add_shadow(image, corners):

    if random.random() < 0.40:
        return image

    height, width = image.shape
    shadow = np.zeros((height, width), dtype=np.uint8)

    offset = np.float32([
        random.randint(2, 10),
        random.randint(2, 10),
    ])
    shadow_corners = (corners + offset).astype(np.int32)
    cv2.fillConvexPoly(shadow, shadow_corners, 255)
    shadow = cv2.GaussianBlur(shadow, (0, 0), random.uniform(3, 10))

    darkness = random.randint(8, 45)
    result = image.astype(np.float32)
    result -= shadow.astype(np.float32) / 255 * darkness
    return np.clip(result, 0, 255).astype(np.uint8)

def change_lighting(image):

    if random.random() >= 0.65:
        return image

    height, width = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if random.random() < 0.5:

        start = (
            random.randint(-width // 2, width // 2),
            random.randint(-height // 3, height),
        )
        end = (
            random.randint(width // 2, width + width // 2),
            random.randint(0, height + height // 3),
        )
        cv2.line(
            mask,
            start,
            end,
            255,
            random.randint(height // 6, height // 2),
        )
    else:

        cv2.ellipse(
            mask,
            (random.randrange(width), random.randrange(height)),
            (
                random.randint(width // 5, width),
                random.randint(height // 8, height // 2),
            ),
            random.randint(0, 180),
            0,
            360,
            255,
            -1,
        )

    mask = cv2.GaussianBlur(mask, (0, 0), random.uniform(6, 20))
    strength = random.randint(35, 100)
    direction = random.choice([-1, 1])

    result = image.astype(np.float32)
    result += direction * mask.astype(np.float32) / 255 * strength
    return np.clip(result, 0, 255).astype(np.uint8)

def add_photo_effects(image):

    contrast = random.uniform(0.75, 1.25)
    brightness = random.randint(-25, 25)
    image = (image.astype(np.float32) - 127) * contrast + 127 + brightness
    image = np.clip(image, 0, 255).astype(np.uint8)

    if random.random() < 0.70:
        image = cv2.GaussianBlur(
            image,
            random.choice([(3, 3), (5, 5)]),
            random.uniform(0.3, 2.0),
        )

    if random.random() < 0.20:
        length = random.choice([5, 7, 9, 11])
        kernel = np.zeros((length, length), dtype=np.float32)

        if random.random() < 0.5:
            kernel[length // 2, :] = 1
        else:
            np.fill_diagonal(kernel, 1)

        kernel /= kernel.sum()
        image = cv2.filter2D(image, -1, kernel)

    noise = np.random.normal(0, random.uniform(0.5, 3.5), image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return image

def generate_image(task):

    (
        index,
        settings,
        output_dir,
        outside_count,
        size_group,
        has_objects,
        image_format,
        orientation,
        perspective_group,
    ) = task

    random.seed(settings["seed"] + index)
    np.random.seed(settings["seed"] + index)

    settings = settings.copy()
    image_width, image_height = random_image_size(settings, image_format)
    settings["image_width"] = image_width
    settings["image_height"] = image_height

    paper_width, paper_height = random_paper_size(settings)
    settings["paper_width"] = paper_width
    settings["paper_height"] = paper_height

    source = np.float32([
        [0, 0],
        [paper_width - 1, 0],
        [paper_width - 1, paper_height - 1],
        [0, paper_height - 1],
    ])

    corners, fallback_used = random_paper_position(
        settings,
        outside_count,
        size_group,
        orientation,
        perspective_group,
    )
    matrix = cv2.getPerspectiveTransform(source, corners)
    image = create_background(settings)
    paper = create_paper(settings)

    hard_object_on_top = has_objects and random.random() < 0.40

    if has_objects:
        if not hard_object_on_top:
            image = add_large_object(image, corners)

        if random.random() < 0.45:
            image = add_small_objects(image, minimum=1, maximum=2)

    image = add_shadow(image, corners)

    cv2.warpPerspective(
        paper,
        matrix,
        (settings["image_width"], settings["image_height"]),
        dst=image,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    if hard_object_on_top:
        image = add_large_object(image, corners)

    if has_objects and random.random() < 0.55:
        image = add_small_objects(image, minimum=1, maximum=2)

    if has_objects and random.random() < 0.35:
        image = add_cable(image, corners)

    image = change_lighting(image)
    image = add_photo_effects(image)
    image_path = Path(output_dir) / f"img_{index:06d}.jpg"
    cv2.imwrite(
        str(image_path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, random.randint(45, 95)],
    )

    labelled_corners = order_corners(corners)
    return (
        labelled_corners,
        (image_width, image_height),
        image_format,
        fallback_used,
    )

def generate_dataset(settings):

    output_dir = Path(__file__).parent / settings["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if list(output_dir.glob("img_*.jpg")):
        raise FileExistsError(
            "Папка dataset_generated уже содержит изображения"
        )

    image_count = settings["image_count"]
    random_generator = random.Random(settings["seed"])

    full_count = int(image_count * 0.50)
    one_outside_count = int(image_count * 0.20)
    two_outside_count = image_count - full_count - one_outside_count
    outside_cases = (
        [0] * full_count
        + [1] * one_outside_count
        + [2] * two_outside_count
    )

    small_count = int(image_count * 0.15)
    medium_count = int(image_count * 0.25)
    large_count = image_count - small_count - medium_count
    size_groups = (
        ["small"] * small_count
        + ["medium"] * medium_count
        + ["large"] * large_count
    )

    without_objects_count = image_count // 2
    object_cases = (
        [False] * without_objects_count
        + [True] * (image_count - without_objects_count)
    )

    portrait_count = int(image_count * 0.50)
    square_count = int(image_count * 0.20)
    image_formats = (
        ["portrait"] * portrait_count
        + ["square"] * square_count
        + ["landscape"] * (image_count - portrait_count - square_count)
    )

    usual_count = image_count // 3
    strong_count = image_count // 3
    perspective_groups = (
        ["usual"] * usual_count
        + ["strong"] * strong_count
        + ["extreme"] * (image_count - usual_count - strong_count)
    )

    random_generator.shuffle(outside_cases)
    random_generator.shuffle(size_groups)
    random_generator.shuffle(object_cases)
    random_generator.shuffle(image_formats)
    random_generator.shuffle(perspective_groups)

    orientations = [None] * image_count
    for index in range(image_count):
        is_full_large = (
            outside_cases[index] == 0
            and size_groups[index] == "large"
        )
        if is_full_large and image_formats[index] == "landscape":
            orientations[index] = "horizontal"
        elif is_full_large and image_formats[index] == "portrait":
            orientations[index] = "other"

    horizontal_count = int(image_count * 0.60)
    already_horizontal = orientations.count("horizontal")
    free_indices = [
        index
        for index, value in enumerate(orientations)
        if value is None
    ]
    random_generator.shuffle(free_indices)
    remaining_horizontal = horizontal_count - already_horizontal
    for index in free_indices[:remaining_horizontal]:
        orientations[index] = "horizontal"
    for index in free_indices[remaining_horizontal:]:
        orientations[index] = "other"

    corners_list = []
    image_sizes = []
    saved_formats = []
    fallback_flags = []
    tasks = (
        (
            index,
            settings,
            str(output_dir),
            outside_cases[index],
            size_groups[index],
            object_cases[index],
            image_formats[index],
            orientations[index],
            perspective_groups[index],
        )
        for index in range(image_count)
    )

    with ProcessPoolExecutor(
        max_workers=settings["worker_count"],
    ) as executor:

        results = executor.map(generate_image, tasks, chunksize=10)

        for index, result in enumerate(results):
            corners, image_size, image_format, fallback_used = result
            corners_list.append(corners)
            image_sizes.append(image_size)
            saved_formats.append(image_format)
            fallback_flags.append(fallback_used)

            if (index + 1) % 5000 == 0:
                print("Создано:", index + 1)

    np.savez_compressed(
        output_dir / "corners.npz",
        corners=np.asarray(corners_list, dtype=np.float32),
        image_sizes=np.asarray(image_sizes, dtype=np.int32),
        image_formats=np.asarray(saved_formats),
        paper_orientations=np.asarray(orientations),
        perspective_groups=np.asarray(perspective_groups),
        outside_corner_counts=np.asarray(outside_cases, dtype=np.uint8),
        size_groups=np.asarray(size_groups),
        has_objects=np.asarray(object_cases, dtype=np.bool_),
        fallback_used=np.asarray(fallback_flags, dtype=np.bool_),
    )
    return output_dir

if __name__ == "__main__":
    dataset_dir = generate_dataset(generator_settings)
    print("Датасет сохранён:", dataset_dir)
