import cv2
import numpy as np
from anomalyBlue import finalizeMask, preprocessRoi, runAnomalyPipeline

LOWER_BLUE = np.array([22, 20, 150], dtype=np.uint8)
UPPER_BLUE = np.array([225, 120, 255], dtype=np.uint8)


def segmentBlueMaskYcbcr(
    roi_bgr,
    lower_blue,
    upper_blue,
    min_fragment_area,
    min_b=55,
    min_b_minus_r=30,
    min_b_minus_g=12,
    min_cb_minus_cr=34,
    min_cb_y_ratio=0.22,
):
    processed = preprocessRoi(roi_bgr)
    ycrcb = cv2.cvtColor(processed, cv2.COLOR_BGR2YCrCb)

    y = ycrcb[:, :, 0].astype(np.float32)
    cr = ycrcb[:, :, 1].astype(np.int16)
    cb = ycrcb[:, :, 2].astype(np.int16)
    cb_shift = cb - 128
    cr_neg = 128 - cr

    b = processed[:, :, 0].astype(np.int16)
    g = processed[:, :, 1].astype(np.int16)
    r = processed[:, :, 2].astype(np.int16)

    strict_ycrcb = cv2.inRange(ycrcb, lower_blue, upper_blue) > 0
    strict_chroma = (
        ((cb - cr) >= min_cb_minus_cr)
        & (cb_shift >= (min_cb_y_ratio * y))
        & ((cb_shift * 100) >= (cr_neg * 55))
    )
    strict_bgr = (
        (b >= min_b)
        & ((b - r) >= min_b_minus_r)
        & ((b - g) >= min_b_minus_g)
    )
    strict_raw = np.where(strict_ycrcb & strict_chroma & strict_bgr, 255, 0).astype(np.uint8)

    relaxed_lower = np.array(
        [
            max(0, int(lower_blue[0]) - 6),
            max(0, int(lower_blue[1]) - 5),
            max(128, int(lower_blue[2]) - 8),
        ],
        dtype=np.uint8,
    )
    relaxed_upper = np.array(
        [
            min(255, int(upper_blue[0]) + 10),
            min(127, int(upper_blue[1]) + 4),
            int(upper_blue[2]),
        ],
        dtype=np.uint8,
    )
    relaxed_ycrcb = cv2.inRange(ycrcb, relaxed_lower, relaxed_upper) > 0
    relaxed_chroma = (
        ((cb - cr) >= max(20, min_cb_minus_cr - 10))
        & (cb_shift >= (max(0.14, min_cb_y_ratio - 0.06) * y))
        & ((cb_shift * 100) >= (cr_neg * 48))
    )
    relaxed_bgr = (
        (b >= max(35, min_b - 15))
        & ((b - r) >= max(18, min_b_minus_r - 10))
        & ((b - g) >= max(8, min_b_minus_g - 4))
    )
    relaxed_raw = np.where(relaxed_ycrcb & relaxed_chroma & relaxed_bgr, 255, 0).astype(np.uint8)

    return finalizeMask(strict_raw, relaxed_raw, min_fragment_area)


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
        segmentBlueMaskYcbcr,
        lower_blue,
        upper_blue,
        min_area,
        merge_distance,
        roi,
        draw_mask_overlay,
    )
