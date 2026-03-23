from pathlib import Path
import os
import nd2
from tifffile import imread, imwrite
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from skimage.measure import regionprops_table
import pyclesperanto_prototype as cle
import apoc

cle.select_device("RTX")

def create_results_folders(main_directory_path):

    folders_to_create = []

    # Create results directory
    experiment_id = main_directory_path.stem

    # Create a 'results' folder in the root directory
    results_infection_folder = Path("results") / experiment_id / "sum_infection_results"
    results_per_cell_label_folder = Path("results") / experiment_id / "per_cell_label_results"
    results_per_bacteria_label_folder = Path("results") / experiment_id / "per_bacteria_label_results"

    # Append to the folders to create list
    folders_to_create.append(results_infection_folder)
    folders_to_create.append(results_per_cell_label_folder)
    folders_to_create.append(results_per_bacteria_label_folder)

    for path in folders_to_create:
        try:
            os.makedirs(path)
            print(f"'{path}' folder created successfully.")
        except FileExistsError:
            print(f"'{path}' folder already exists.")

    return results_infection_folder, results_per_cell_label_folder, results_per_bacteria_label_folder

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
        print(f"\nImage analyzed: {filename}")
        #print(f"Original Array shape: {img.shape}")
        #print(f"Compressed Array shape: {img.shape}")

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

def count_points_in_labels(points, cell_labels):
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
    cell_labels : ndarray, shape (H, W)
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
    H, W = cell_labels.shape
    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x, y = x[valid], y[valid]

    # Get label IDs at point locations
    labels_at_points = cell_labels[y, x]

    # Count efficiently with bincount
    max_label = cell_labels.max()
    counts = np.bincount(labels_at_points, minlength=max_label + 1)

    # Build DataFrame (skip background label 0)
    label_ids = np.nonzero(counts)[0]
    num_points = counts[label_ids]

    counts_df = pd.DataFrame({
        "label": label_ids,
        "num_points": num_points
    })

    return counts_df[counts_df["label"] != 0].reset_index(drop=True)

def brightfield_correction(directory_path, images, slicing_factor_xy):
    """Substract uneven and remove background from BF by obtaining the median of all BF channels"""

    try:

        bf_correction = imread(directory_path / "bf_correction.tiff")
        print("Brightfield correction image found.")

    except FileNotFoundError:

        print("Generating brightfield correction image...")

        # Create an empty list to store the brightfield images from each well
        bf_arrays = []

        # Read all images, extract the brightfield channel and calculate the mean to correct illumination and remove dust spots
        for image in tqdm(images):

            # Read image, apply slicing if needed and return filename and img as a np array
            img, filename = read_image(image, slicing_factor_xy, log=False)

            # Extract brighfield slice
            bf_channel = img[4]

            # Add it to the bf_arrays iterable
            bf_arrays.append(bf_channel)

        # Create a stack containing all bf images
        bf_stack = np.stack(bf_arrays, axis=0)

        # Calculate the median to retain the common structures (spots, illumination)
        bf_correction = np.median(bf_stack, axis=0)
        del bf_stack

        # Store brightfield correction as .tiff to avoid recalculating it everytime
        imwrite(directory_path / "bf_correction.tiff",bf_correction)

    return bf_correction

def detect_infected_cells(img, mtb_segmenter, cell_labels, plate_nr, well_id, infection_stats):
        """Detect infected cells and subcellular compartments"""
        print("\nDetecting infected cells and subcellular compartments...")
        # Generate membrane and cytoplasm labels from cell_labels
        membrane_labels = cle.reduce_labels_to_label_edges(cell_labels)
        membrane_labels = cle.pull(membrane_labels)
        cytoplasm_labels = cle.erode_labels(cell_labels, radius=1)
        cytoplasm_labels = cle.pull(cytoplasm_labels)

        # Detect Mtb spots
        mtb_labels = mtb_segmenter.predict(img[3])
        mtb_labels = cle.pull(mtb_labels)

        # Convert mtb_labels to boolean mask
        mtb_boolean = mtb_labels.astype(bool)

        # Use NumPy's indexing to identify cell labels that intersect with mtb_boolean (bacterial mask)
        infected_cell_labels = np.unique(cell_labels[mtb_boolean])
        infected_cell_labels = infected_cell_labels[infected_cell_labels != 0]

        # Use NumPy's indexing to identify membrane labels that intersect with mtb_boolean (bacterial mask)
        infected_membrane_labels = np.unique(membrane_labels[mtb_boolean])
        infected_membrane_labels = infected_membrane_labels[infected_membrane_labels != 0]

        # Use NumPy's indexing to identify cytoplasm labels that intersect with mtb_boolean (bacterial mask)
        infected_cytoplasm_labels = np.unique(cytoplasm_labels[mtb_boolean])
        infected_cytoplasm_labels = infected_cytoplasm_labels[infected_cytoplasm_labels != 0]

        # Extract stats for each cell and subcellular compartment
        infected_cell_mask = np.isin(cell_labels, infected_cell_labels)
        non_infected_cell_mask = np.isin(cell_labels, infected_cell_labels, invert=True)
        infected_cells_array = np.where(infected_cell_mask, cell_labels, 0).astype(cell_labels.dtype)
        non_infected_cells_array = np.where(non_infected_cell_mask, cell_labels, 0).astype(cell_labels.dtype)

        infected_membrane_mask = np.isin(membrane_labels, infected_membrane_labels)
        non_infected_membrane_mask = np.isin(membrane_labels, infected_membrane_labels, invert=True)
        infected_membranes_array = np.where(infected_membrane_mask, membrane_labels, 0).astype(membrane_labels.dtype)
        non_infected_membranes_array = np.where(non_infected_membrane_mask, membrane_labels, 0).astype(membrane_labels.dtype)

        infected_cytoplasm_mask = np.isin(cytoplasm_labels, infected_cytoplasm_labels)
        non_infected_cytoplasm_mask = np.isin(cytoplasm_labels, infected_cytoplasm_labels, invert=True)
        infected_cytoplasms_array = np.where(infected_cytoplasm_mask, cytoplasm_labels, 0).astype(cytoplasm_labels.dtype)
        non_infected_cytoplasms_array = np.where(non_infected_cytoplasm_mask, cytoplasm_labels, 0).astype(cytoplasm_labels.dtype)

        infected_cells = len(np.unique(infected_cells_array)) - (0 in infected_cells_array)
        non_infected_cells = len(np.unique(non_infected_cells_array)) - (0 in non_infected_cells_array)
        total_cells = cell_labels.max()

        infected_membranes = len(np.unique(infected_membranes_array)) - (0 in infected_membranes_array)
        non_infected_membranes = len(np.unique(non_infected_membranes_array)) - (0 in non_infected_membranes_array)
        total_membranes = len(np.unique(membrane_labels[membrane_labels != 0]))

        infected_cytoplasms = len(np.unique(infected_cytoplasms_array)) - (0 in infected_cytoplasms_array)
        non_infected_cytoplasms = len(np.unique(non_infected_cytoplasms_array)) - (0 in non_infected_cytoplasms_array)
        total_cytoplasms = len(np.unique(cytoplasm_labels[cytoplasm_labels != 0]))

        # Calculate percentage of infected cells / membrane / cytoplasm regions
        perc_inf_cells = round(infected_cells / total_cells * 100, 2) if total_cells > 0 else 0
        perc_inf_membranes = round(infected_membranes / total_membranes * 100, 2) if total_membranes > 0 else 0
        perc_inf_cytoplasms = round(infected_cytoplasms / total_cytoplasms * 100, 2) if total_cytoplasms > 0 else 0
        # Share of segmented cells with Mtb in cytoplasm or on membrane (same denominator as whole-cell infection)
        perc_cytoplasm_inf_cells = (round(infected_cytoplasms / total_cells * 100, 2) if total_cells > 0 else 0)
        perc_membrane_inf_cells = (round(infected_membranes / total_cells * 100, 2) if total_cells > 0 else 0)

        print(f"Total cells: {total_cells}")
        print(f"Percentage infected:{perc_inf_cells}")

        # Create a dictionary containing all extracted info per image
        stats_dict = {
                    "plate": plate_nr,
                    "well_id": well_id,
                    "total_nr_cells": total_cells,
                    "infected_cells": infected_cells,
                    "non-infected_cells": non_infected_cells,
                    "%_inf_cells": perc_inf_cells,
                    "total_nr_membranes": total_membranes,
                    "infected_membranes": infected_membranes,
                    "non-infected_membranes": non_infected_membranes,
                    "%_inf_membranes": perc_inf_membranes,
                    "total_nr_cytoplasms": total_cytoplasms,
                    "infected_cytoplasms": infected_cytoplasms,
                    "non-infected_cytoplasms": non_infected_cytoplasms,
                    "%_inf_cytoplasms": perc_inf_cytoplasms,
                    "%_cytoplasm_inf_cells": perc_cytoplasm_inf_cells,
                    "%_membrane_inf_cells": perc_membrane_inf_cells,
        }

        # Append the current data point to the stats_list
        infection_stats.append(stats_dict)

        return mtb_labels, membrane_labels, cytoplasm_labels, infected_cell_labels, infected_membrane_labels, infected_cytoplasm_labels

def extract_mtb_regionprops(mtb_labels, plate_nr, well_id, image):

    print("Extracting Mtb properties...")

    # Single list of regionprops to request (modify based on needs)
    regionprops_properties = [
        "label",
        "area",                          # number of voxels (volume in voxel units)
        "axis_major_length",             # length of major axis from inertia tensor (elongation)
        "axis_minor_length",             # length of minor axis (second principal axis in 3D)
        "equivalent_diameter_area",      # diameter of sphere with same volume as region
        "euler_number",                  # topology: objects + holes − tunnels (connectivity)
        "extent",                        # volume / bounding-box volume (fill of the box)
        "feret_diameter_max",            # maximum Feret (caliper) diameter
        "solidity"                      # volume / convex-hull volume (compact vs lobed)
    ]

    # Create a dictionary containing all image metadata
    descriptor_dict = {
        "plate": plate_nr,
        "well_id": well_id,
        "filepath": image
        }

    # Extract morphological features from bacterial labels (mtb_labels)
    props = regionprops_table(label_image=mtb_labels,
                        properties=regionprops_properties)

    # Convert to dataframe
    props_df = pd.DataFrame(props)

    # Add each key-value pair from descriptor_dict to props_df at the specified position
    insertion_position = 0
    for key, value in descriptor_dict.items():
        props_df.insert(insertion_position, key, value)
        insertion_position += 1  # Increment position to maintain the order of keys in descriptor_dict

    return props_df

def map_bacterial_location(mtb_labels, cell_labels, membrane_labels, cytoplasm_labels, mtb_props_df):
    """Map mtb_labels to different locations: out of cell, cell, membrane, cytoplasm"""

    # Convert cell and subcellular labels to boolean mask
    cell_boolean = cell_labels.astype(bool)
    membrane_boolean = membrane_labels.astype(bool)
    cytoplasm_boolean = cytoplasm_labels.astype(bool)

    # Use Numpy's indexing to identify mtb_labels that intersect with the different locations
    mtb_labels_in_cell = np.unique(mtb_labels[cell_boolean])
    mtb_labels_in_cell = mtb_labels_in_cell[mtb_labels_in_cell != 0] # Drop background 
    mtb_labels_in_cytoplasm = np.unique(mtb_labels[cytoplasm_boolean])
    mtb_labels_in_cytoplasm = mtb_labels_in_cytoplasm[mtb_labels_in_cytoplasm != 0] # Drop background 
    mtb_labels_in_membrane = np.unique(mtb_labels[membrane_boolean])
    mtb_labels_in_membrane = mtb_labels_in_membrane[mtb_labels_in_membrane != 0] # Drop background

    # Reverse logic for mtb_labels outside of cell
    all_mtb = np.unique(mtb_labels)
    all_mtb = all_mtb[all_mtb != 0]
    mtb_labels_outside = np.setdiff1d(all_mtb, mtb_labels_in_cell)

    # Rename label(id) to CellProfiler format ObjectNumber
    mtb_props_df.rename(columns={"label": "ObjectNumber"}, inplace=True)

    # Add infected flags to props_df based on cellular location
    # Find position of "ObjectNumber" column
    col_idx = mtb_props_df.columns.get_loc("ObjectNumber")

    # Insert new column right after "ObjectNumber"
    mtb_props_df.insert(col_idx + 1, "location_cell", mtb_props_df["ObjectNumber"].isin(mtb_labels_in_cell))
    mtb_props_df.insert(col_idx + 2, "location_cytoplasm", mtb_props_df["ObjectNumber"].isin(mtb_labels_in_cytoplasm))
    mtb_props_df.insert(col_idx + 3, "location_membrane", mtb_props_df["ObjectNumber"].isin(mtb_labels_in_membrane))
    mtb_props_df.insert(col_idx + 4, "location_extracellular", mtb_props_df["ObjectNumber"].isin(mtb_labels_outside))
    
    return mtb_props_df

def extract_cell_features(img, cell_labels, markers, plate_nr, well_id, image):

    print("Extracting per marker intensity information...")
    # Define features to extract
    regionprops_properties = [
    "label",
    "area",                          # number of voxels (volume in voxel units)
    "area_bbox",                     # volume of axis-aligned bounding box
    "area_convex",                   # volume of convex hull of the region
    "area_filled",                   # volume after filling holes
    "axis_major_length",             # length of major axis from inertia tensor (elongation)
    "axis_minor_length",             # length of minor axis (second principal axis in 3D)
    "equivalent_diameter_area",      # diameter of sphere with same volume as region
    "euler_number",                  # topology: objects + holes − tunnels (connectivity)
    "extent",                        # volume / bounding-box volume (fill of the box)
    "feret_diameter_max",            # maximum Feret (caliper) diameter
    "solidity",                      # volume / convex-hull volume (compact vs lobed)
    "inertia_tensor_eigvals",        # eigenvalues of inertia tensor (3 values: shape/orientation)
    "intensity_mean",
    "intensity_min",
    "intensity_max",
    "intensity_std",
    ]

    # Empty list to populate with per channel intensity information
    props_list = []

    # Create a dictionary containing all image metadata
    descriptor_dict = {
        "plate": plate_nr,
        "well_id": well_id,
        "filepath": image
        }
    for channel_name, ch_nr in markers:

        # Extract intensity information from each channel
        props = regionprops_table(label_image=cell_labels,
                        intensity_image=img[ch_nr],
                        properties=regionprops_properties)
        
        # Convert to dataframe
        props_df = pd.DataFrame(props)
        
        # Rename intensity columns to human readable format
        props_df.rename(columns={"intensity_mean": f"Intensity_MeanIntensity_Cytoplasm_{channel_name}_ch{ch_nr}"}, inplace=True)
        props_df.rename(columns={"intensity_max": f"Intensity_MaxIntensity_Cytoplasm_{channel_name}_ch{ch_nr}"}, inplace=True)
        props_df.rename(columns={"intensity_min": f"Intensity_MinIntensity_Cytoplasm_{channel_name}_ch{ch_nr}"}, inplace=True)
        props_df.rename(columns={"intensity_std": f"Intensity_StdIntensity_Cytoplasm_{channel_name}_ch{ch_nr}"}, inplace=True)

        # Append each props_df to props_list
        props_list.append(props_df)

    # Initialize the df with the first df in the list
    props_df = props_list[0]
    # Start looping from the second df in the list
    for df in props_list[1:]:
        props_df = props_df.merge(df, on="label")

    # Add each key-value pair from descriptor_dict to props_df at the specified position
    insertion_position = 0
    for key, value in descriptor_dict.items():
        props_df.insert(insertion_position, key, value)
        insertion_position += 1  # Increment position to maintain the order of keys in descriptor_dict

    return props_df

def puncta_detection(img, puncta_markers, spotiflow_model, cell_labels, props_df):

    print("Detecting spots in puncta markers...")
    for puncta_marker in puncta_markers:

        # Load PixelClassifier for the corresponding puncta marker
        puncta_cl_filename =f"./pretrained_classifiers/no_nuclei_signal/siMtb screen I_LØ/{puncta_marker[0]}_segmenter.cl"
        puncta_segmenter = apoc.PixelClassifier(opencl_filename=puncta_cl_filename)

        # Obtaining Spotiflow predicted points 
        points, details = spotiflow_model.predict(img[puncta_marker[1]], subpix=True, verbose=False)

        # Defining puncta signal mask
        mask = puncta_segmenter.predict(img[puncta_marker[1]])

        # Filter the predicted Spotiflow points intersecting with puncta mask
        filtered_points = filter_points_interpolated(points, mask)

        # Count how many points per cell
        puncta_counts_df = count_points_in_labels(filtered_points, cell_labels)

        # Add marker name to num_points column name
        puncta_counts_df.rename(columns={"num_points": f"{puncta_marker[0]}_num_points"}, inplace=True)

        #Merge puncta counts with the props_df containing intensity information
        props_df = props_df.merge(
            puncta_counts_df,
            on="label",
            how="left"   # keep all labels from props_df
        )

        #Fill missing counts with zero in current puncta marker column
        props_df[f"{puncta_marker[0]}_num_points"] = props_df[f"{puncta_marker[0]}_num_points"].fillna(0).astype(int)

    return props_df