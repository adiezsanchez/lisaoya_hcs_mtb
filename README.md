# Lisaoya HCS-MTB — High-content screening analysis of *M. tuberculosis* infection in 2D cell cultures

A GPU-accelerated pipeline for automated analysis of multichannel 2D fluorescence images from a high-content screening (HCS) assay acquired on a Nikon Crestoptics V3 spinning-disk confocal. The workflow segments cells with CellposeSAM, detects *Mtb* bacteria (using [APOC ObjectSegmenter](https://github.com/haesleinhuepf/apoc)) and autophagy-related puncta (LC3B, GAL3, Chmp4B - using [Spotiflow](https://github.com/weigertlab/spotiflow)), quantifies per-cell and per-bacterium features, and exports plate-level CSV summaries for downstream heatmaps and exploratory plots.

<h2> Data acquisition and file naming conventions </h2>

> [!WARNING]
> - A **CUDA-capable GPU** is required. Notebooks raise an error if Cellpose cannot access the GPU.
> - Pipeline is designed to run on Windows (win-64).
> - In `utils.py` and the analysis notebooks, `cle.select_device("RTX")` must match your OpenCL device name (edit if needed).
> - Plate folders whose names contain `Nuc` or `Results` are excluded from batch processing (nuclei-stain-only runs).

> [!NOTE]
> - Input images are read as **Nikon `.nd2`**. Stacks are expected in channel-first order after loading: `(channel, Y, X)`.
> - The batch pipeline assumes the channel layout below. If your acquisition order differs, update the `markers` and `puncta_markers` lists in the notebooks and retrain classifiers as needed.

**Expected channel indices (0-based)**

| Index | Marker  | Optical config (example) | Role in pipeline                          |
|------:|---------|--------------------------|-------------------------------------------|
| 0     | LC3B    | SD_AF647                 | Fluorescence + puncta (Spotiflow + APOC)  |
| 1     | GAL3    | SD_RFP                   | Fluorescence + puncta (Spotiflow + APOC)  |                    
| 2     | Chmp4B  | SD_GFP                   | Fluorescence + puncta (Spotiflow + APOC)  |                    
| 3     | Mtb     | SD_DAPI                  | Bacterial detection (APOC ObjectSegmenter)|
| 4     | BF      | SD_BF                    | Brightfield correction & CellposeSAM input|

**Recommended folder layout**

```text
<experiment_root>/                    # e.g. siMtb screen I_LØ
├── Plate01/                          # one subdirectory per 96-well plate
│   ├── Plate01_..._Wells-A1__....nd2
│   ├── Plate01_..._Wells-A2__....nd2
│   └── ...
├── Plate02/
└── ...
```

Well metadata is parsed from filenames: `plate_nr` from the segment before the first `_`, and `well_id` from the substring after `Wells-` (e.g. `A1`, `H12`).

Training data for APOC classifiers (optional, when retraining) should follow:

```text
train_data/no_nuclei_signal/<experiment_id>/
├── multichannel_crops/               # multichannel TIFF crops
├── SD_AF647_LC3B_detection_training/
│   ├── images/
│   └── annotations/
├── SD_RFP_GAL3_detection_training/
├── SD_GFP_Chmp4B_detection_training/
└── SD_DAPI_Mtb_detection_training/
```

Use `pixi run split_multichannel` (or `python scripts/split_multichannel_training_data.py`) to split multichannel crops into per-marker training folders.

<h2> How to install this tool? (Environment setup) </h2>

> [!TIP]
> In order to run these Jupyter notebooks and scripts you will need to familiarize yourself with Python virtual environments, IDEs, and Git. If you are not familiar with those concepts, watch the [Before you start (Python, IDE and Git on Windows)](https://youtu.be/tzdFuxF2E3U) video, it covers the basics for running pipelines like this one on Windows.
>
> **TL;DR** You are busy in the wet lab, skip to the **Pixi** section below.

| | Watch on YouTube | Description |
|-------|------------------|-------------|
| | [Before you start (Python, IDE and Git on Windows)](https://youtu.be/tzdFuxF2E3U) | Python, virtual environments, IDEs, and Git on Windows — configure your machine before running the notebooks |
| | [Pipeline installation using Pixi](https://youtu.be/tzdFuxF2E3U) | **TL;DR** Quick path to a reproducible environment with Pixi |

1. Clone this repository:

   ```bash
   git clone https://github.com/adiezsanchez/lisaoya_hcs_mtb
   ```

2. If you do not have Git installed, download the code as a `.zip` from the green **Code** button on GitHub.

3. Proceed with **Pixi** as the recommended environment manager.

> [!TIP]
> [Pixi](https://pixi.sh/latest/installation/) provides a reproducible environment from `pixi.toml` and `pixi.lock`. This project is currently pinned to **`win-64`** (Windows + NVIDIA CUDA 12.1).

**Install Pixi (Windows PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

Close and reopen your terminal, then install dependencies and launch Jupyter Lab:

```bash
cd lisaoya_hcs_mtb && pixi install && pixi run lab
```

Other useful tasks:

| Task | Command |
|------|---------|
| Jupyter Lab | `pixi run lab` |
| Napari viewer | `pixi run napari` |
| Split multichannel training TIFFs | `pixi run split_multichannel` |

**Legacy conda/mamba environment**

A frozen `environment.yml` (`hcs_cellpose`) is also provided for mamba/conda users. Core packages match the Pixi definition (Python 3.10, Cellpose 4, PyTorch 2.5 + CUDA 12.1, Spotiflow, nd2, napari, APOC). After creating the env, install Spotiflow from PyPI if needed: `pip install spotiflow`.

<h2> Workflow summary </h2>

> [!IMPORTANT]
> - Set `main_directory_path` in the analysis notebooks to your experiment root on disk.
> - Optional XY downsampling: set `slicing_factor_xy` to `2` or `4` in the batch/single notebooks (default `None` = full resolution).
> - Brightfield correction is cached per plate as `bf_correction.tiff` in the plate folder (median of all BF channels).
> - Pretrained APOC classifiers ship under `pretrained_classifiers/no_nuclei_signal/siMtb screen I_LØ/`. Retrain with the `0_*_detection_training.ipynb` notebooks if markers or imaging conditions change.

**Training notebooks (APOC + Spotiflow QC)**

| Notebook | Purpose |
|----------|---------|
| `0_Mtb_detection_training.ipynb` | Train APOC **ObjectSegmenter** for *Mtb* (channel 3) |
| `0_AF647_LC3B_detection_training.ipynb` | Train APOC **PixelClassifier** puncta mask for LC3B; Spotiflow filtering demos |
| `0_RFP_Gal3_detection_training.ipynb` | Train APOC **PixelClassifier** puncta mask for GAL3 |

**Notebook 1: Single well QC (`1_SP_Single_Image_visualization.ipynb`)**

- Runs the full per-image pipeline on one `.nd2` well for interactive QC in Napari.
- CellposeSAM segmentation on summed LC3B+GAL3 and BF-corrected brightfield.
- *Mtb* detection, subcellular infection flags, per-cell intensities, bacterial load, and puncta counts.
- Layers for input channels, labels, and Spotiflow points.

**Notebook 2: Batch plate processing (`1_BP_Batch_Processing_Plate_analysis.ipynb`)**

- Loops over all plate subfolders (excluding `Nuc` / `Results`) and all wells in each plate.
- Writes three CSV types per plate under `results/<experiment_id>/` (see [CSV outputs](#csv-outputs) below).

**Notebook 3: Exploratory data analysis (`2_Exploratory_Data_analysis.ipynb`)**

- Loads per-cell CSVs and explores relationships between puncta counts, marker intensity, and infection status (scatter, KDE, correlation).

**Notebook 4: Plate heatmaps (`2_Plate_plot_generation.ipynb`)**

- Generates 96-well heatmaps (infection %, cell counts, puncta in infected vs non-infected cells, *Mtb* morphology, compartment-resolved bacterial counts) via `utils_data_analysis.plot_plate_view`.

<h2>Analysis pipeline (per well)</h2>

1. **Read** `.nd2` image; optional XY downsampling.
2. **Brightfield correction** — median BF stack across the plate; cache `bf_correction.tiff`.
3. **Cell segmentation** — CellposeSAM on `[LC3B+GAL3, BF − correction]`.
4. ***Mtb* segmentation** — APOC ObjectSegmenter on channel 3; derive membrane and cytoplasm masks from cell labels (pyclesperanto).
5. **Infection statistics** — per-well counts and percentages of infected cells, membranes, and cytoplasms.
6. **Per-bacterium features** — `regionprops` morphology + compartment flags (`location_cell`, `location_cytoplasm`, `location_membrane`, `location_extracellular`).
7. **Per-cell features** — morphology and mean/min/max/std intensity per marker channel.
8. **Bacterial load** — `mtb_area_sum` and `%_bacterial_load` per cell.
9. **Puncta** — Spotiflow point detection per marker, filtered by APOC puncta masks; `{marker}_num_points` per cell.

Core logic lives in `utils.py`; plate plotting helpers are in `utils_data_analysis.py`.

<h2> CSV outputs </h2>

Results are written under `results/<experiment_id>/`:

| Subfolder | File pattern | Content |
|-----------|--------------|---------|
| `sum_infection_results/` | `results_infection_<plate>.csv` | One row per well: infection counts and percentages |
| `per_cell_label_results/` | `results_per_label_<plate>.csv` | One row per segmented cell (all wells concatenated) |
| `per_bacteria_label_results/` | `mtb_results_per_label_<plate>.csv` | One row per *Mtb* object |

<details>
<summary><b>Key output columns</b></summary>

**Per-well infection summary (`sum_infection_results`)**

- `plate`, `well_id`
- `total_nr_cells`, `infected_cells`, `non-infected_cells`, `%_inf_cells`
- `total_nr_membranes`, `infected_membranes`, `non-infected_membranes`, `%_inf_membranes`
- `total_nr_cytoplasms`, `infected_cytoplasms`, `non-infected_cytoplasms`, `%_inf_cytoplasms`
- `%_cytoplasm_inf_cells`, `%_membrane_inf_cells`

**Per-cell table (`per_cell_label_results`)**

- `plate`, `well_id`, `filepath`, `ObjectNumber` (cell label ID)
- `Mtb_infected_cell`, `Mtb_infected_membrane`, `Mtb_infected_cytoplasm`
- Morphology: `area`, `area_bbox`, `area_convex`, `area_filled`, `axis_major_length`, `axis_minor_length`, `equivalent_diameter_area`, `euler_number`, `extent`, `feret_diameter_max`, `solidity`
- Intensity (per marker): `Intensity_MeanIntensity_Cytoplasm_<marker>_ch<n>`, plus `_Max`, `_Min`, `_Std` variants
- Load: `mtb_area_sum`, `%_bacterial_load`
- Puncta: `LC3B_num_points`, `GAL3_num_points`, `Chmp4B_num_points`

**Per-bacterium table (`per_bacteria_label_results`)**

- `plate`, `well_id`, `filepath`, `ObjectNumber`
- `location_cell`, `location_cytoplasm`, `location_membrane`, `location_extracellular`
- Morphology: `area`, `axis_major_length`, `axis_minor_length`, `equivalent_diameter_area`, `euler_number`, `extent`, `feret_diameter_max`, `solidity`

</details>

<h2> Materials and Methods: Image Analysis </h2>

2D multichannel `.nd2` images from a 96-well HCS assay were analyzed in Python. Cells were segmented with **CellposeSAM** (Cellpose 4) using a two-channel input: the sum of LC3B and GAL3 fluorescence plus brightfield after plate-level median correction. *Mycobacterium* regions were segmented with an **APOC ObjectSegmenter** trained on the *Mtb* channel. Membrane and cytoplasm compartments were derived from cell labels via label-edge reduction and erosion (pyclesperanto). Infection was scored when bacterial labels overlapped whole cells, eroded cytoplasm, or membrane masks (on a cell compartment basis).

Per-cell marker intensities and standard morphological descriptors were extracted with `skimage.measure.regionprops_table`. Per-bacterium morphology and boolean compartment assignments were computed similarly. **Bacterial load** per cell was defined as the number of *Mtb* foreground pixels overlapping the cell mask, divided by cell `area` (%). **Puncta** were detected with **Spotiflow** (`general` pretrained model), then filtered to regions predicted by marker-specific **APOC PixelClassifiers** (interpolated mask value ≥ 1.8). Puncta counts per cell were obtained by mapping Spotiflow coordinates to the cell segmentation.

Batch results were aggregated per plate into infection-summary, per-cell, and per-bacterium CSV files for plate-view heatmaps and exploratory statistics.

<h2> How to cite this pipeline </h2>

If you use this pipeline in your work, please cite the repository and contact the maintainer for an archived DOI when available.

- For licensing terms, see `LICENSE` (BSD 3-Clause).

<h2> Related publications </h2>

Placeholder for publications citing this pipeline.
