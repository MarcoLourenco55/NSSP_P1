"""
Structural Preprocessing Pipeline
Includes skull stripping (BET) and tissue segmentation (FAST)
"""

import os
import glob
from pathlib import Path
from nipype.interfaces.fsl import FAST


# -------------------------------------------------------------------
# Skull Stripping (BET)
# -------------------------------------------------------------------
def skull_strip(input_file, output_file, robust=False, threshold=0.2):
    """
    Perform skull stripping on a given MRI image using FSL's BET tool.

    Parameters
    ----------
    input_file : str
        Path to the input T1-weighted MRI file.
    output_file : str
        Path for saving the skull-stripped output.
    robust : bool
        Whether to use robust center estimation (-R flag in BET). Default is False.
    threshold : float
        Fractional intensity threshold (between 0 and 1). Lower values give larger brain outline.
    """
    cmd = f"bet {input_file} {output_file} -m -f {threshold} {'-R' if robust else ''}"
    print(f"Running: {cmd}")
    os.system(cmd)
    print(f"Skull stripping completed: {output_file}")


# -------------------------------------------------------------------
# Tissue Segmentation (FAST)
# -------------------------------------------------------------------
def run_fast_segmentation(bet_path, output_dir):
    """
    Apply FAST segmentation to a skull-stripped brain image.

    Parameters
    ----------
    bet_path : str
        Path to the brain-extracted image (output from BET).
    output_dir : str
        Directory where the segmentation results will be saved.

    Returns
    -------
    str
        Path to the segmentation output directory.
    """
    if not os.path.exists(bet_path):
        raise FileNotFoundError(f"Input file not found: {bet_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Clean up old FAST files
    for f in glob.glob(os.path.join(output_dir, '*fast*')):
        os.remove(f)

    fast = FAST()
    fast.inputs.in_files = bet_path
    fast.inputs.out_basename = os.path.join(output_dir, 'T1w_fast_segmentation')
    fast.inputs.args = '-v'

    print("Running FAST segmentation...")
    fast.run()
    print(f"Segmentation completed: {output_dir}")
    return output_dir


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main(project_root: Path = None):
    """
    Structural preprocessing pipeline combining:
      1. Skull stripping using BET
      2. Tissue segmentation using FAST
    """

    # Automatically find project root if not provided
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]  # two levels up
        print(f"Detected project root: {project_root}")

    # Respect the folder structure of the main pipeline
    t1_input = project_root / "Preprocessing" / "Structural" / "T1w.nii.gz"
    skullstrip_dir = project_root / "Preprocessing" / "Structural" / "Skull striping"
    segmentation_dir = project_root / "Preprocessing" / "Structural" / "Segmentation"

    skullstrip_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    bet_output = skullstrip_dir / "T1w_bet.nii.gz"

    # Step 1: Skull stripping
    print("=== Step 1: Skull Stripping (BET) ===")
    skull_strip(str(t1_input), str(bet_output), robust=False, threshold=0.2)

    # Step 2: Segmentation
    print("\n=== Step 2: Tissue Segmentation (FAST) ===")
    run_fast_segmentation(str(bet_output), str(segmentation_dir))

    print("\nStructural preprocessing completed successfully.")


if __name__ == "__main__":
    main()
