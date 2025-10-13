from tifffile import imwrite, imread
from pathlib import Path
import os 

def list_images(directory_path, format=None):
    """
    List image files in a given directory.

    Parameters
    ----------
    directory_path : Path
        Path to the folder containing image files.
    format : str, optional
        Specific file format to filter by (e.g., "tif").
        If not provided, the function will look for .czi and .nd2 files.

    Returns
    -------
    images : list of str
        List of full file paths for all matching images.
    """

    # Create an empty list to store all image file paths within the dataset directory
    images = []

    # If a specific format is defined, only include files with that extension
    if format:
        for file_path in directory_path.glob(f"*.{format}"):
            images.append(str(file_path))
    else:
        # Otherwise, collect files with known microscopy formats (.czi, .nd2)
        for file_path in directory_path.glob("*.czi"):
            images.append(str(file_path))
        for file_path in directory_path.glob("*.nd2"):
            images.append(str(file_path))

    return images


# Define the mapping between biological markers, optical configurations, and channel indices
# Format: (marker_name, optical_configuration, channel_number)
markers = [
    ("LC3B", "SD_AF647", 0),
    ("GAL3", "SD_RFP", 1),
    ("Chmp4B", "SD_GFP", 2),
    ("Mtb", "SD_DAPI", 3)
]

# Define the root directory where input and output data are located
directory_path = Path("./train_data/no_nuclei_signal/siMtb screen I_LØ")

# Define the subfolder containing the original multichannel TIFF crops
input_directory = directory_path / "multichannel_crops"
print(f"Splitting multichannel files in: {input_directory}")

# List all input TIFF images
images = list_images(input_directory, format="tif")

# Loop through each multichannel image in the dataset
for image in images:
    
    # Read the multichannel TIFF file into a NumPy array
    img = imread(image)
    
    # Extract the filename without extension (used for saving)
    filename = Path(image).stem
    
    # Iterate over all defined markers to extract the corresponding channel
    for marker, oc, channel_nr in markers:
        
        # Define the output directory for this specific marker and optical configuration
        save_directory = directory_path / f"{oc}_{marker}_detection_training" / "images"

        # Create empty folder for manual annotations
        annotations_directory = directory_path / f"{oc}_{marker}_detection_training" / "annotations"
        
        # Create the directories if they do not exist already
        try:
            os.makedirs(save_directory)
        except FileExistsError:
            pass

        try:
            os.makedirs(annotations_directory)
        except FileExistsError:
            pass
        
        # Extract the (x, y) 2D image corresponding to the current channel
        output_image = img[channel_nr]
        
        # Save the single-channel image as a new TIFF in the appropriate folder
        imwrite(save_directory / f"{filename}.tif", output_image)

print("Split multichannel completed")