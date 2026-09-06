import math

import cv2
import numpy as np

def order_corners(points):
    points = np.float32(points).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(angles)]
    start = np.argmin(points[:, 0] + points[:, 1])
    return np.roll(points, -start, axis=0)

def warp_document(image, points, output_size=(600, 850)):
    width, height = output_size
    destination = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])
    matrix = cv2.getPerspectiveTransform(order_corners(points), destination)
    return cv2.warpPerspective(image, matrix, (width, height))

def draw_corners(image, points, color=(0, 0, 255)):
    result = image.copy()
    points = order_corners(points).astype(np.int32)
    cv2.polylines(result, [points], True, color, 4)
    for point in points:
        cv2.circle(result, tuple(point), 7, color, -1)
    return result

def analyze_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        5,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphology = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )
    edges = cv2.Canny(morphology, 50, 150)

    height, width = edges.shape
    found = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        60,
        minLineLength=int(min(width, height) * 0.15),
        maxLineGap=int(min(width, height) * 0.05),
    )

    data = {
        "binary": binary,
        "morphology": morphology,
        "edges": edges,
        "segments": [],
        "selected_lines": [],
        "angle1": None,
        "angle2": None,
        "corners": None,
        "problem": "",
    }

    if found is None:
        data["problem"] = "Преобразование Хафа не нашло линии"
        return data

    lines = []
    for x1, y1, x2, y2 in found.reshape(-1, 4):
        line_length = math.hypot(x2 - x1, y2 - y1)
        line_angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        lines.append([x1, y1, x2, y2, line_length, line_angle])

    lines.sort(key=lambda line: line[4], reverse=True)
    lines = lines[:40]
    data["segments"] = [line[:4] for line in lines]

    line_angles = [line[5] for line in lines]
    line_lengths = [line[4] for line in lines]

    bins = np.arange(0, 190, 10)
    scores, _ = np.histogram(line_angles, bins=bins, weights=line_lengths)
    best = int(np.argmax(scores))
    angle1 = (bins[best] + bins[best + 1]) / 2
    angle2 = (angle1 + 90) % 180
    data["angle1"] = angle1
    data["angle2"] = angle2

    pairs = []
    for needed_angle in (angle1, angle2):
        suitable = []
        for line in lines:
            difference = abs(line[5] - needed_angle) % 180
            if min(difference, 180 - difference) < 25:
                suitable.append(line)

        if len(suitable) < 2:
            data["problem"] = "Не удалось выбрать четыре границы листа"
            return data

        radians = math.radians(needed_angle)
        normal_x = -math.sin(radians)
        normal_y = math.cos(radians)

        positions = []
        for line in suitable:
            middle_x = (line[0] + line[2]) / 2
            middle_y = (line[1] + line[3]) / 2
            position = middle_x * normal_x + middle_y * normal_y
            positions.append((position, line[:4]))

        positions.sort(key=lambda item: item[0])
        pairs.append((positions[0][1], positions[-1][1]))

    data["selected_lines"] = list(pairs[0]) + list(pairs[1])
    corners = []

    for first in pairs[0]:
        for second in pairs[1]:
            x1, y1, x2, y2 = map(float, first)
            x3, y3, x4, y4 = map(float, second)
            a1, b1, c1 = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
            a2, b2, c2 = y3 - y4, x4 - x3, x3 * y4 - x4 * y3
            determinant = a1 * b2 - a2 * b1

            if abs(determinant) < 1e-6:
                data["problem"] = "Выбранные линии не пересекаются"
                return data

            x = (b1 * c2 - b2 * c1) / determinant
            y = (c1 * a2 - c2 * a1) / determinant
            corners.append([x, y])

    data["corners"] = order_corners(corners)
    return data

def detect_corners_hough(image):
    result = analyze_hough(image)
    if result["corners"] is None:
        raise ValueError(result["problem"])
    return result["corners"]
