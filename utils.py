from pathlib import Path
import nd2
import tifffile
import napari
import numpy as np
import os
import pandas as pd
from skimage import measure
from scipy.ndimage import map_coordinates
import pyclesperanto_prototype as cle

cle.select_device("RTX")

def list_images (directory_path, format=None):

    # Create an empty list to store all image filepaths within the dataset directory
    images = []

    # If manually defined format
    if format:
        for file_path in directory_path.glob(f"*.{format}"):
            images.append(str(file_path))

    else:
        # Iterate through the .czi and .nd2 files in the directory
        for file_path in directory_path.glob("*.czi"):
            images.append(str(file_path))
            
        for file_path in directory_path.glob("*.nd2"):
            images.append(str(file_path))

    return images

def read_image (image, slicing_factor_xy, log=True):
    """Read raw image microscope files (.nd2), apply downsampling if needed and return filename and a numpy array"""

    # Read path storing raw image and extract filename
    file_path = Path(image)
    filename = file_path.stem

    # Extract file extension
    extension = file_path.suffix

    if extension == ".nd2":
        # Read stack from .nd2 (z, ch, x, y) or (ch, x, y)
        img = nd2.imread(image)
        
    else:
        print ("Implement new file reader")

    # Apply slicing trick to reduce image size (xy resolution)
    try:
        img = img[:, ::slicing_factor_xy, ::slicing_factor_xy]
    except IndexError as e:
        print(f"Slicing Error: {e}")
        print(f"Slicing parameters: Slicing_XY:{slicing_factor_xy}")

    if log:
        # Feedback for researcher
        print(f"\n\nImage analyzed: {filename}")
        print(f"Original Array shape: {img.shape}")
        print(f"Compressed Array shape: {img.shape}")

    return img, filename

def filter_points_interpolated(points, predicted_labels, threshold=1.8):
    """
    Keep points that fall inside mask == 2 (with subpixel precision).
    Uses bilinear interpolation and a threshold.

    Coordinate system:
    ------------------
    - Spotiflow outputs points as (row, col) = (y, x),
      which matches NumPy array indexing.
    - predicted_labels is indexed as [row, col] = [y, x].
    - map_coordinates also expects coords in (row, col) order.
    - In Napari a point is y pixels down from the top and x pixels to the right starting from the left edge.

        (0,0) ─────────►  x (width, cols = second axis)
          │
          │
          ▼
          y (height, rows = first axis)

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Input points in (row, col) = (y, x) order.
    predicted_labels : ndarray, shape (H, W)
        Mask array. Regions with value==2 are of interest.
    threshold : float, optional
        Interpolated value threshold to keep a point.
        Default 1.8 keeps values close to 2.

    Returns
    -------
    filtered_points : ndarray, shape (M, 2)
        Subset of input points that lie within mask==2
        according to subpixel interpolation.
    """
    H, W = predicted_labels.shape

    # Spotiflow already gives (y, x), which matches map_coordinates
    coords = np.vstack([points[:, 0], points[:, 1]])

    # Interpolate values at float coords (order=1 → bilinear)
    values = map_coordinates(predicted_labels, coords, order=1, mode="nearest")

    # Keep points where interpolated value is close enough to 2
    keep = values >= threshold
    return points[keep]

def count_points_in_labels(points, cytoplasm_labels):
    """
    Count how many Spotiflow points fall inside each cytoplasm label.

    Coordinate system:
    ------------------
    - Spotiflow outputs (row, col) = (y, x).
    - NumPy arrays are indexed as [row, col] = [y, x].
    - We therefore swap when unpacking, so that:
        x = col = points[:,1]
        y = row = points[:,0]

        (0,0) ─────────►  x (width, cols)
          │
          │
          ▼
          y (height, rows)

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Spotiflow points in (row, col) = (y, x) order.
    cytoplasm_labels : ndarray, shape (H, W)
        Label image with integer IDs for each segmented cell.

    Returns
    -------
    counts_df : pd.DataFrame
        Table with columns: [label, num_points].
    """
    if len(points) == 0:
        return pd.DataFrame(columns=["label", "num_points"])

    # Convert to integer pixel indices
    points_int = np.floor(points).astype(int)
    x, y = points_int[:, 1], points_int[:, 0]  # swap for [y, x] indexing

    # Keep only points inside image bounds
    H, W = cytoplasm_labels.shape
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y = x[valid], y[valid]

    # Get label IDs at point locations
    labels_at_points = cytoplasm_labels[y, x]

    # Count efficiently with bincount
    max_label = cytoplasm_labels.max()
    counts = np.bincount(labels_at_points, minlength=max_label + 1)

    # Build DataFrame (skip background label 0)
    label_ids = np.nonzero(counts)[0]
    num_points = counts[label_ids]

    counts_df = pd.DataFrame({
        "label": label_ids,
        "num_points": num_points
    })

    return counts_df[counts_df["label"] != 0].reset_index(drop=True)


