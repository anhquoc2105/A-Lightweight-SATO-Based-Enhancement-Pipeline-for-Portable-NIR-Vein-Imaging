import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
from skimage.filters import sato


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = BASE_DIR / "image" / "1.png"
DEFAULT_SCALE_PERCENT = 50
BACKGROUND_THRESHOLD = 100
BACKGROUND_CLOSE_SIZE = 15
BACKGROUND_OPEN_SIZE = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Xu ly anh ban tay va hien thi mask mach mau."
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"Duong dan anh input. Mac dinh: {DEFAULT_IMAGE_PATH}",
    )
    return parser.parse_args()


def read_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")
    return image


def resize_image(image, scale_percent=DEFAULT_SCALE_PERCENT):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def remove_background(image, threshold=BACKGROUND_THRESHOLD):
    """Mask grayscale input; also accept BGR for existing helper callers."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, hand_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    hand_mask = cv2.morphologyEx(
        hand_mask,
        cv2.MORPH_CLOSE,
        np.ones((BACKGROUND_CLOSE_SIZE, BACKGROUND_CLOSE_SIZE), np.uint8),
    )
    hand_mask = cv2.morphologyEx(
        hand_mask,
        cv2.MORPH_OPEN,
        np.ones((BACKGROUND_OPEN_SIZE, BACKGROUND_OPEN_SIZE), np.uint8),
    )
    binary_mask = (hand_mask > 0).astype(np.uint8)
    background_removed_image = image * (
        binary_mask if image.ndim == 2 else binary_mask[:, :, np.newaxis]
    )
    return background_removed_image, binary_mask


def create_inner_hand_mask(hand_mask):
    mask_uint8 = (hand_mask * 255).astype(np.uint8)
    distance = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    inner_mask = np.zeros_like(mask_uint8)
    inner_mask[distance > 12] = 255
    return inner_mask


def enhance_grayscale(gray_image):
    """Apply first-stage CLAHE to an already grayscale, background-masked image."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_image)
    return gray_image, enhanced_gray


def apply_sato_filter(gray_image):
    filtered = sato(
        gray_image,
        sigmas=range(1, 10),
        black_ridges=True,
        mode="constant",
        cval=0,
    )
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    if filtered.max() > 0:
        filtered = filtered / filtered.max()
    return np.uint8(filtered * 255)


def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def preserve_edges(image):
    return cv2.edgePreservingFilter(
        image,
        flags=cv2.RECURS_FILTER,
        sigma_s=5,
        sigma_r=0.4,
    )


def phan_doan_lan_can(image, ksize):
    """Compute local thresholds in bulk, preserving the legacy pixel alignment."""
    rows, cols = image.shape
    padding = (ksize - 1) // 2
    padded_image = np.pad(image, (padding, padding), mode="reflect")
    padded_float = padded_image.astype(np.float64)
    integral = cv2.integral(padded_float)
    integral_sq = cv2.integral(padded_float * padded_float)
    y1 = np.arange(rows)
    x1 = np.arange(cols)
    # Clipping also preserves the original truncated windows for even ksize.
    y2 = np.minimum(y1 + ksize, padded_image.shape[0])
    x2 = np.minimum(x1 + ksize, padded_image.shape[1])
    area = (y2 - y1)[:, None] * (x2 - x1)[None, :]

    window_sum = (
        integral[np.ix_(y2, x2)] - integral[np.ix_(y1, x2)]
        - integral[np.ix_(y2, x1)] + integral[np.ix_(y1, x1)]
    )
    window_sum_sq = (
        integral_sq[np.ix_(y2, x2)] - integral_sq[np.ix_(y1, x2)]
        - integral_sq[np.ix_(y2, x1)] + integral_sq[np.ix_(y1, x1)]
    )
    local_mean = window_sum / area
    variance = np.maximum(window_sum_sq / area - local_mean * local_mean, 0)
    threshold = 20 * np.sqrt(variance) + np.mean(padded_image)
    # Keep the legacy top-left comparison; centering it would change the masks.
    return np.where(padded_image[:rows, :cols] > threshold, 255, 0).astype(np.uint8)


def clean_vein_mask(segmented_mask):
    mask_uint8 = segmented_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    opened_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened_mask,
        connectivity=8,
    )
    clean_mask = np.zeros_like(opened_mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area > 50:
            clean_mask[labels == label] = 255

    return clean_mask


def resize_mask_to_image(mask, image):
    if image.shape[:2] == mask.shape:
        return mask
    return cv2.resize(mask, (image.shape[1], image.shape[0]))


def keep_inner_veins(vein_mask, inner_hand_mask):
    vein_mask = cv2.bitwise_and(
        vein_mask.astype(np.uint8),
        inner_hand_mask.astype(np.uint8),
    )
    vein_mask = cv2.morphologyEx(
        vein_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    vein_mask = morphology.remove_small_objects(
        vein_mask.astype(bool),
        min_size=40,
        connectivity=2,
    )
    return vein_mask.astype(np.uint8) * 255


def clear_bottom_border(vein_mask, border_rows=3):
    if border_rows < 0:
        raise ValueError("border_rows must be greater than or equal to 0")

    result = vein_mask.copy()
    if border_rows > 0:
        result[-min(border_rows, result.shape[0]) :, :] = 0
    return result


def create_vein_overlay(hand_image, vein_mask):
    red_layer = np.zeros_like(hand_image)
    red_layer[vein_mask == 255] = [0, 0, 255]
    return cv2.addWeighted(hand_image, 1.0, red_layer, 0.95, 0)


def process_image(image_path):
    original_image = read_image(image_path)
    resized_image = resize_image(original_image)
    raw_gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray_image, hand_mask = remove_background(raw_gray_image)
    # Preserve the BGR diagnostic output used by existing batch scripts.
    background_removed_image = resized_image * hand_mask[:, :, np.newaxis]
    inner_hand_mask = create_inner_hand_mask(hand_mask)

    gray_image, enhanced_gray = enhance_grayscale(gray_image)
    sato_image = apply_sato_filter(enhanced_gray)
    contrast_image = enhance_contrast(sato_image)
    edge_preserved_image = preserve_edges(contrast_image)

    resized_nir = cv2.resize(edge_preserved_image, (300, 300))
    vein_mask = phan_doan_lan_can(resized_nir, ksize=3)
    vein_mask = cv2.resize(vein_mask, (500, 500))
    vein_mask = clean_vein_mask(vein_mask)
    vein_mask = resize_mask_to_image(vein_mask, background_removed_image)
    inner_hand_mask = resize_mask_to_image(inner_hand_mask, background_removed_image)
    vein_mask = keep_inner_veins(vein_mask, inner_hand_mask)
    vein_mask = clear_bottom_border(vein_mask)
    clahe_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    overlay_image = create_vein_overlay(clahe_bgr, vein_mask)

    return {
        "resized_image": resized_image,
        "raw_gray_image": raw_gray_image,
        "background_removed_image": background_removed_image,
        "gray_image": gray_image,
        "enhanced_gray": enhanced_gray,
        "sato_image": sato_image,
        "contrast_image": contrast_image,
        "edge_preserved_image": edge_preserved_image,
        "vein_mask": vein_mask,
        "overlay_image": overlay_image,
    }


def show_gray(image, title, position):
    plt.subplot(position)
    plt.title(title)
    plt.imshow(image, cmap="gray")
    plt.axis("off")


def show_bgr(image, title, position):
    plt.subplot(position)
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")


def show_results(results):
    plt.figure()
    show_bgr(results["resized_image"], "Raw Image", 331)
    show_gray(results["raw_gray_image"], "Grayscale", 332)
    show_gray(results["gray_image"], "Background Removal", 333)
    show_gray(results["enhanced_gray"], "CLAHE 1", 334)
    show_gray(results["sato_image"], "SATO", 335)
    show_gray(results["contrast_image"], "CLAHE 2", 336)
    show_gray(results["edge_preserved_image"], "Edge-Preserving", 337)
    show_gray(results["vein_mask"], "Clean Vein Mask", 338)
    show_bgr(results["overlay_image"], "Vein Overlay", 339)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    results = process_image(args.file)
    show_results(results)


if __name__ == "__main__":
    main()
