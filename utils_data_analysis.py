import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_plate_view(df, column_name, plate, title, label, fmt=3, display=True, cmap="magma"):
    # --- Parse well_id into row (A–H) and column (1–12) ---
    def split_well_id(well):
        match = re.match(r"([A-H])(\d{1,2})", str(well))
        if match:
            row, col = match.groups()
            return row, int(col)
        return None, None

    df[["row", "col"]] = df["well_id"].apply(lambda x: pd.Series(split_well_id(x)))

    # --- Pivot into 96-well plate layout ---
    plate_matrix = df.pivot(index="row", columns="col", values=column_name)

    # Reindex rows and columns to enforce full plate structure
    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    plate_matrix = plate_matrix.reindex(index=rows, columns=cols)

    # --- Plot heatmap ---
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        plate_matrix,
        cmap=cmap,  # or "coolwarm", "magma" etc.
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={'label': label},
        annot=True, fmt=f".{fmt}f"
    )

    plt.title(f"{plate} - {title}", fontsize=14)
    plt.xlabel("Column")
    plt.ylabel("Row")

    # Rotate row (y-axis) labels 90° to the right
    ax.set_yticklabels(ax.get_yticklabels(), rotation=-90, va="center")

    # --- Save plot ---
    save_dir = f"./results/plate_view/{column_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plate}_{column_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if display:
        plt.show()
    else:
        plt.close()

    print(f"Saved plate view to {save_path}")


