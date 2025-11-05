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
    # Construct the BET command
    cmd = f"bet {input_file} {output_file} -m -f {threshold} {'-R' if robust else ''}"
    print(f"Running: {cmd}")
    os.system(cmd)
    print("Skull stripping completed successfully.")


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
    # Check if the input file exists
    if not os.path.exists(bet_path):
        raise FileNotFoundError(f"Input file not found: {bet_path}")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Clean up old FAST files in the output directory
    for f in glob.glob(os.path.join(output_dir, '*fast*')):
        os.remove(f)

    # Initialize FAST segmentation
    fast = FAST()
    fast.inputs.in_files = bet_path
    fast.inputs.out_basename = os.path.join(output_dir, 'T1w_fast_segmentation')
    fast.inputs.args = '-v'  # Verbose mode for detailed logs

    print("Running FAST segmentation...")
    fast.run()

    print(f"Segmentation completed. Results saved in: {output_dir}")
    return output_dir

def main():
    """
    Structural preprocessing pipeline combining:
      1. Skull stripping using BET
      2. Tissue segmentation using FAST
    """
    
    project_root = Path("/Users/amadeus/Documents/Neural Signal Processing/NSSPProj1")
    out_dir_BET = project_root / "Skull striping"
    out_dir_SEG = project_root / "Segmentation"
    out_dir_BET.mkdir(parents=True, exist_ok=True)
    out_dir_SEG.mkdir(parents=True, exist_ok=True)
    

    # Define input and output paths
    t1_path = project_root / "NSSP/T1w/T1w.nii.gz" 
    bet_output =  out_dir_BET / "T1w_bet.nii.gz"
    segmentation_dir = out_dir_SEG / "T1w_fast_segmentation"

    # Step 1: Skull stripping
    print("Step 1: Skull Stripping (BET)")
    skull_strip(str(t1_path), str(bet_output), robust=False, threshold=0.2)

    # Step 2: Segmentation (FAST)
    print("\nStep 2: Tissue Segmentation (FAST)")
    run_fast_segmentation(str(bet_output), str(segmentation_dir))

    print("\nStructural preprocessing completed successfully.")


if __name__ == "__main__":
    main()
