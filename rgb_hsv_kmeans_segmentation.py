"""
RGB and HSV Image Segmentation using a custom K-means implementation.

This project compares image segmentation performed in RGB and HSV color spaces.
Initial centroids are selected with a greedy farthest-point strategy.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def greedy_centroid_selection(pixels: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """
    Select initial centroids using a greedy farthest-point strategy.

    The first centroid is selected randomly. Each subsequent centroid is
    selected as the pixel with the largest distance from its nearest
    already-selected centroid.
    """
    if pixels.ndim != 2:
        raise ValueError("pixels must be a 2D array.")
    if len(pixels) < n_clusters:
        raise ValueError("Number of pixels must be at least the number of clusters.")

    centroids = [pixels[rng.integers(len(pixels))]]

    for _ in range(1, n_clusters):
        distances = np.min(
            np.linalg.norm(
                pixels[:, None] - np.asarray(centroids)[None, :],
                axis=2,
            ),
            axis=1,
        )
        centroids.append(pixels[np.argmax(distances)])

    return np.asarray(centroids)


def assign_clusters(pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each pixel to its nearest centroid using Euclidean distance."""
    distances = np.linalg.norm(
        pixels[:, None] - centroids[None, :],
        axis=2,
    )
    return np.argmin(distances, axis=1)


def update_centroids(
    pixels: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """
    Update centroids using the mean of each cluster.

    Empty clusters keep their previous centroid instead of being replaced
    with a zero vector.
    """
    new_centroids = centroids.copy()

    for cluster_id in range(n_clusters):
        cluster_points = pixels[labels == cluster_id]
        if len(cluster_points) > 0:
            new_centroids[cluster_id] = cluster_points.mean(axis=0)

    return new_centroids


def kmeans_segment(
    pixels: np.ndarray,
    image_shape: tuple[int, int, int],
    n_clusters: int = 4,
    max_iter: int = 8,
    seed: int = 42,
) -> np.ndarray:
    """
    Segment an image represented by a pixel matrix using custom K-means.

    Returns an image whose pixels are replaced by their assigned cluster
    centroids.
    """
    rng = np.random.default_rng(seed)

    centroids = greedy_centroid_selection(pixels, n_clusters, rng)

    for _ in range(max_iter):
        labels = assign_clusters(pixels, centroids)
        new_centroids = update_centroids(
            pixels,
            labels,
            centroids,
            n_clusters,
        )

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Recompute labels once using the final centroids.
    labels = assign_clusters(pixels, centroids)

    return centroids[labels].reshape(image_shape).astype(np.uint8)


def save_comparison(
    original_rgb: np.ndarray,
    segmented_rgb: np.ndarray,
    segmented_hsv_rgb: np.ndarray,
    difference: np.ndarray,
    output_dir: Path,
) -> None:
    """Save individual results and a combined comparison figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(output_dir / "RGB_Segmentation.jpg"),
        cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    cv2.imwrite(
        str(output_dir / "HSV_Segmentation.jpg"),
        cv2.cvtColor(segmented_hsv_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    cv2.imwrite(
        str(output_dir / "Segmentation_Difference.jpg"),
        cv2.cvtColor(difference, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    axes[0].imshow(segmented_rgb)
    axes[0].set_title("RGB Segmentation")
    axes[0].axis("off")

    axes[1].imshow(segmented_hsv_rgb)
    axes[1].set_title("HSV Segmentation")
    axes[1].axis("off")

    axes[2].imshow(difference)
    axes[2].set_title("Difference")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(
        output_dir / "Segmentation_Comparison.jpg",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RGB and HSV image segmentation using custom K-means."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory where output images will be saved (default: results).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=4,
        help="Number of K-means clusters (default: 4).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=8,
        help="Maximum number of K-means iterations (default: 8).",
    )
    parser.add_argument(
        "--resize-factor",
        type=float,
        default=0.3,
        help="Resize factor for the input image (default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible centroid initialization (default: 42).",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.clusters < 1:
        raise ValueError("--clusters must be at least 1.")

    if args.max_iter < 1:
        raise ValueError("--max-iter must be at least 1.")

    if not 0 < args.resize_factor <= 1:
        raise ValueError("--resize-factor must be between 0 and 1.")

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not decode image: {args.image}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(
        image_rgb,
        None,
        fx=args.resize_factor,
        fy=args.resize_factor,
        interpolation=cv2.INTER_AREA,
    )

    height, width, channels = image_rgb.shape
    pixels_rgb = image_rgb.reshape((-1, channels))

    segmented_rgb = kmeans_segment(
        pixels_rgb,
        image_rgb.shape,
        n_clusters=args.clusters,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    pixels_hsv = image_hsv.reshape((-1, 3))

    segmented_hsv = kmeans_segment(
        pixels_hsv,
        image_hsv.shape,
        n_clusters=args.clusters,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    # Convert HSV result back to RGB for visualization.
    segmented_hsv_rgb = cv2.cvtColor(segmented_hsv, cv2.COLOR_HSV2RGB)

    # Pixel-wise visual difference between the two segmentation outputs.
    difference = cv2.absdiff(segmented_rgb, segmented_hsv_rgb)

    print(f"Image size after resizing: {width} x {height}")
    print(f"Clusters: {args.clusters}")
    print(f"Maximum iterations: {args.max_iter}")
    print(f"Random seed: {args.seed}")

    save_comparison(
        image_rgb,
        segmented_rgb,
        segmented_hsv_rgb,
        difference,
        args.output_dir,
    )

    print(f"Results saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
