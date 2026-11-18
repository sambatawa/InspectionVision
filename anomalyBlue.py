import cv2
import numpy as np

LOWER_BLUE = np.array([98, 105, 55], dtype=np.uint8)
UPPER_BLUE = np.array([128, 255, 255], dtype=np.uint8)

PASS_COLOR = (255, 255, 255)
FAIL_COLOR = (0, 0, 255)


def drawStatusBanner(img, status, detection_count):
    color = FAIL_COLOR if status == "FAIL" else PASS_COLOR
    title = (
        "FAIL (" + str(detection_count) + " objek)"
        if status == "FAIL"
        else "PASS"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, th1), _ = cv2.getTextSize(title, font, 0.7, 2)
    cv2.putText(img, title, (26, 22 + th1), font, 0.7, color, 2, cv2.LINE_AA)


def drawSegmentationLabel(img, detection):
    x, y, w, _ = detection["bbox"]
    label = "FAIL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (lw, lh), baseline = cv2.getTextSize(label, font, 0.5, 1)

    cx = x + w // 2
    label_x = max(0, min(cx - (lw + 8) // 2, img.shape[1] - lw - 10))
    if y - lh - baseline - 10 >= 0:
        bg_y1 = y - lh - baseline - 10
        bg_y2 = y - 2
        text_y = y - baseline - 6
    else:
        bg_y1 = y + 2
        bg_y2 = y + lh + baseline + 10
        text_y = y + lh + 6
    cv2.rectangle(img, (label_x, bg_y1), (label_x + lw + 8, bg_y2), FAIL_COLOR, -1)
    cv2.putText(img, label, (label_x + 4, text_y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def resolveRoi(frame_shape, roi):
    h, w = frame_shape[:2]
    if roi is None:
        return 0, 0, w, h

    x1, y1, x2, y2 = roi
    if all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)) and any(
        isinstance(v, float) for v in (x1, y1, x2, y2)
    ):
        x1 = int(round(x1 * w))
        y1 = int(round(y1 * h))
        x2 = int(round(x2 * w))
        y2 = int(round(y2 * h))
    else:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def preprocessRoi(roi_bgr):
    denoised = cv2.medianBlur(roi_bgr, 3)

    max_ch = np.max(denoised, axis=2)
    min_ch = np.min(denoised, axis=2)
    chroma = max_ch - min_ch
    neutral_mask = (chroma < 40) & (max_ch >= 50) & (max_ch <= 240)

    neutral_count = int(np.count_nonzero(neutral_mask))
    if neutral_count < int(0.05 * denoised.shape[0] * denoised.shape[1]):
        return denoised

    mean_bgr = np.mean(denoised[neutral_mask].astype(np.float32), axis=0)
    gray_ref = float(np.mean(mean_bgr))
    if gray_ref < 1.0:
        return denoised

    gains = np.clip(gray_ref / np.maximum(mean_bgr, 1.0), 0.85, 1.18)
    return np.clip(denoised.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def applyHysteresis(strict_mask, relaxed_mask, min_seed_area):
    if cv2.countNonZero(strict_mask) == 0:
        return np.zeros_like(strict_mask)

    combined = cv2.bitwise_or(relaxed_mask, strict_mask)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    result = np.zeros_like(combined)
    if num_labels <= 1:
        return result

    seed_labels = labels[strict_mask > 0]
    if seed_labels.size == 0:
        return result

    seed_counts = np.bincount(seed_labels, minlength=num_labels)
    valid_labels = np.where(seed_counts >= min_seed_area)[0]
    for label_idx in valid_labels:
        if label_idx > 0:
            result[labels == label_idx] = 255
    return result


def fillInternalHoles(mask, max_hole_area=350):
    filled = mask.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return filled

    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] != -1 and cv2.contourArea(contour) <= max_hole_area:
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
    return filled


def finalizeMask(strict_raw, relaxed_raw, min_fragment_area):
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    strict_clean = cv2.morphologyEx(strict_raw, cv2.MORPH_OPEN, kernel_open)
    strict_clean = cv2.bitwise_and(strict_clean, strict_raw)

    relaxed_clean = cv2.morphologyEx(relaxed_raw, cv2.MORPH_OPEN, kernel_open)
    relaxed_clean = cv2.bitwise_and(relaxed_clean, relaxed_raw)

    min_seed_area = max(40, min_fragment_area // 2)
    hysteresis_mask = applyHysteresis(strict_clean, relaxed_clean, min_seed_area)
    hysteresis_mask = fillInternalHoles(hysteresis_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hysteresis_mask, connectivity=8)
    filtered_mask = np.zeros_like(hysteresis_mask)
    for label_idx in range(1, num_labels):
        if stats[label_idx, cv2.CC_STAT_AREA] >= min_fragment_area:
            filtered_mask[labels == label_idx] = 255

    return filtered_mask, strict_clean


def segmentBlueMask(
    roi_bgr,
    lower_blue,
    upper_blue,
    min_fragment_area,
    min_b=60,
    min_b_minus_r=32,
    min_b_minus_g=14,
    max_lab_b=114,
):
    processed = preprocessRoi(roi_bgr)
    hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
    lab_b = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)[:, :, 2]

    b = processed[:, :, 0].astype(np.int16)
    g = processed[:, :, 1].astype(np.int16)
    r = processed[:, :, 2].astype(np.int16)

    strict_hsv = cv2.inRange(hsv, lower_blue, upper_blue) > 0
    strict_bgr = (
        (b >= min_b)
        & ((b - r) >= min_b_minus_r)
        & ((b - g) >= min_b_minus_g)
        & ((r * 100) <= (b * 72))
    )
    strict_raw = np.where(strict_hsv & strict_bgr & (lab_b <= max_lab_b), 255, 0).astype(np.uint8)

    relaxed_lower = np.array(
        [
            max(0, int(lower_blue[0]) - 4),
            max(0, int(lower_blue[1]) - 28),
            max(0, int(lower_blue[2]) - 18),
        ],
        dtype=np.uint8,
    )
    relaxed_upper = np.array(
        [
            min(179, int(upper_blue[0]) + 4),
            int(upper_blue[1]),
            int(upper_blue[2]),
        ],
        dtype=np.uint8,
    )
    relaxed_hsv = cv2.inRange(hsv, relaxed_lower, relaxed_upper) > 0
    relaxed_bgr = (
        (b >= max(35, min_b - 15))
        & ((b - r) >= max(18, min_b_minus_r - 10))
        & ((b - g) >= max(8, min_b_minus_g - 5))
        & ((r * 100) <= (b * 80))
    )
    relaxed_raw = np.where(relaxed_hsv & relaxed_bgr & (lab_b <= (max_lab_b + 5)), 255, 0).astype(np.uint8)

    return finalizeMask(strict_raw, relaxed_raw, min_fragment_area)


def extractBlueObjects(
    blue_mask,
    strict_mask,
    offset_xy,
    min_area,
    merge_distance,
    min_thickness_radius=3.0,
):
    x1, y1 = offset_xy

    if merge_distance > 0:
        ksize = merge_distance if merge_distance % 2 == 1 else merge_distance + 1
        kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        grouped_mask = cv2.dilate(blue_mask, kernel_merge)
    else:
        grouped_mask = blue_mask

    group_contours, _ = cv2.findContours(grouped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    valid_mask = np.zeros_like(blue_mask)

    for contour in group_contours:
        gx, gy, gw, gh = cv2.boundingRect(contour)
        roi_group = np.zeros((gh, gw), dtype=np.uint8)
        cv2.drawContours(roi_group, [contour], -1, 255, thickness=cv2.FILLED, offset=(-gx, -gy))

        roi_blue = cv2.bitwise_and(blue_mask[gy:gy + gh, gx:gx + gw], roi_group)
        blue_area = int(cv2.countNonZero(roi_blue))
        if blue_area < min_area:
            continue

        roi_strict = cv2.bitwise_and(strict_mask[gy:gy + gh, gx:gx + gw], roi_blue)
        strict_area = int(cv2.countNonZero(roi_strict))
        if strict_area < max(80, int(0.2 * blue_area)):
            continue

        dist_map = cv2.distanceTransform(roi_blue, cv2.DIST_L2, 3)
        if float(dist_map.max()) < min_thickness_radius:
            continue

        rx, ry, rw, rh = cv2.boundingRect(roi_blue)
        x, y, w, h = x1 + gx + rx, y1 + gy + ry, rw, rh

        valid_mask[gy:gy + gh, gx:gx + gw] = cv2.bitwise_or(
            valid_mask[gy:gy + gh, gx:gx + gw],
            roi_blue,
        )
        candidates.append((blue_area, (x, y, w, h)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    detections = [
        {
            "id": idx,
            "bbox": (x, y, w, h),
            "area": float(blue_area),
            "center": (x + w // 2, y + h // 2),
        }
        for idx, (blue_area, (x, y, w, h)) in enumerate(candidates, start=1)
    ]

    return detections, valid_mask


def runAnomalyPipeline(
    imagePath,
    segmentFn,
    lower_blue,
    upper_blue,
    min_area=600,
    merge_distance=35,
    roi=None,
    draw_mask_overlay=True,
):
    img = cv2.imread(imagePath) if isinstance(imagePath, str) else imagePath
    if img is None:
        raise ValueError("Gambar atau frame tidak valid.")

    x1, y1, x2, y2 = resolveRoi(img.shape, roi)
    roi_img = img[y1:y2, x1:x2]

    if roi is not None:
        result_img = np.zeros_like(img)
        result_img[y1:y2, x1:x2] = roi_img
    else:
        result_img = img.copy()

    min_fragment_area = max(100, min_area // 6)
    blue_mask, strict_mask = segmentFn(roi_img, lower_blue, upper_blue, min_fragment_area)

    total_pixels = roi_img.shape[0] * roi_img.shape[1]
    if cv2.countNonZero(blue_mask) < min_area:
        detections = []
        blue_ratio = 0.0
        status = "PASS"
    else:
        detections, valid_mask = extractBlueObjects(
            blue_mask,
            strict_mask,
            (x1, y1),
            min_area,
            merge_distance,
        )
        blue_pixel_count = int(np.count_nonzero(valid_mask))
        blue_ratio = blue_pixel_count / total_pixels
        status = "FAIL" if len(detections) > 0 else "PASS"

        if status == "FAIL":
            if draw_mask_overlay:
                roi_canvas = result_img[y1:y2, x1:x2].copy()
                overlay = roi_canvas.copy()
                overlay[valid_mask > 0] = FAIL_COLOR
                cv2.addWeighted(overlay, 0.55, roi_canvas, 0.45, 0, roi_canvas)
                result_img[y1:y2, x1:x2] = roi_canvas

            for det in detections:
                drawSegmentationLabel(result_img, det)

    if roi is not None and status == "FAIL":
        cv2.rectangle(result_img, (x1, y1), (x2 - 1, y2 - 1), FAIL_COLOR, 2)

    drawStatusBanner(result_img, status, len(detections))
    return status, detections, blue_ratio, result_img


def processBlueAnomaly(
    imagePath,
    lower_blue=LOWER_BLUE,
    upper_blue=UPPER_BLUE,
    min_area=600,
    merge_distance=35,
    roi=None,
    draw_mask_overlay=True,
):
    return runAnomalyPipeline(
        imagePath,
        segmentBlueMask,
        lower_blue,
        upper_blue,
        min_area,
        merge_distance,
        roi,
        draw_mask_overlay,
    )
