#!/usr/bin/env python3
"""
FSL-based functional preprocessing (Part 1.2 style) + optional EPI->T1 coreg

Pipeline:
  1) Per-run: quick mask, global std, rescale to sigma=1
  2) Concatenate runs along time
  3) Motion correction (MCFLIRT)
  4) Gaussian smoothing (FWHM -> sigma)
  5) [Bonus] EPI->T1 coreg for visual report only (NOT used later)

T1 handling:
  - If you pass --t1_brain, it is used directly (no BET).
  - Else if you pass --t1, we run BET with --bet_f/--bet_g.
  - Optionally pass --t1_mask to use your own brain mask (skip BET).

Requirements: FSL on PATH; Python 3.8+
"""

import os, shlex, subprocess, argparse
from pathlib import Path

# ---------- helpers ----------
def run(cmd: str, cwd: Path):
    """Run a shell command with LC_NUMERIC=C to ensure '.' decimal separator."""
    print(f"$ {cmd}")
    env = os.environ.copy(); env["LC_NUMERIC"] = "C"
    subprocess.run(shlex.split(cmd), check=True, cwd=str(cwd), env=env)

def check_cmd(name: str):
    try:
        subprocess.run(["which", name], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise SystemExit(f"[ERROR] '{name}' not found on PATH. Is FSL installed and configured?")

def fslstats_scalar(image: Path, mask: Path, stat: str = "-S") -> float:
    env = os.environ.copy(); env["LC_NUMERIC"] = "C"
    out = subprocess.check_output(
        ["fslstats", str(image), "-k", str(mask), stat], env=env
    ).decode().strip()
    return float(out.split()[0])


def q(p):  
    return shlex.quote(str(p))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="FSL-based functional preprocessing (Part 1.2) + coreg.")
    # Inputs
    ap.add_argument("--lr", required=True, type=Path, help="Path to LR run (e.g., tfMRI_MOTOR_LR.nii)")
    ap.add_argument("--rl", required=True, type=Path, help="Path to RL run (e.g., tfMRI_MOTOR_RL.nii)")
    ap.add_argument("--outdir", type=Path, default=Path("."), help="Output directory (default: .)")
    ap.add_argument("--fwhm", type=float, default=5.0, help="Gaussian smoothing FWHM in mm (default: 5)")
    ap.add_argument("--do_fd", action="store_true", help="Also compute FD/outliers (confounds)")

    # T1 options (choose ONE of --t1_brain or --t1; optional --t1_mask)
    ap.add_argument("--t1_brain", type=Path, default=None, help="Skull-stripped T1 (brain only). Use this if available.")
    ap.add_argument("--t1", type=Path, default=None, help="Raw T1 (we will run BET).")
    ap.add_argument("--t1_mask", type=Path, default=None, help="Optional brain mask for raw T1 (skip BET).")

    # BET tunables (only used if --t1 is given and no --t1_mask)
    ap.add_argument("--bet_f", type=float, default=0.30, help="BET fractional intensity threshold (default 0.30)")
    ap.add_argument("--bet_g", type=float, default=0.0,  help="BET vertical gradient (default 0.0)")

    # Visual extras
    ap.add_argument("--apply_4d_to_t1", action="store_true",
                    help="Apply EPI->T1 xfm to the whole 4D (visual QC only).")

    args = ap.parse_args()

    # Tools
    for tool in ["fslmaths", "fslstats", "bet", "fslmerge", "mcflirt", "flirt", "fslval"]:
        check_cmd(tool)
    if args.do_fd:
        check_cmd("fsl_motion_outliers")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sigma = args.fwhm / 2.355

    # Short names
    LR_mean = outdir/"LR_mean"; LR_mean_brain = outdir/"LR_mean_brain"; LR_mask = outdir/"LR_mean_brain_mask"; LR_var1 = outdir/"LR_var1"
    RL_mean = outdir/"RL_mean"; RL_mean_brain = outdir/"RL_mean_brain"; RL_mask = outdir/"RL_mean_brain_mask"; RL_var1 = outdir/"RL_var1"
    CONCAT = outdir/"subj_concat_var1"; MC = outdir/"subj_concat_var1_mc"
    SMOOTH = outdir/f"subj_concat_var1_mc_s{int(round(args.fwhm))}mm"

    # -----------------------------
    # 1) Per-run variance rescaling
    # -----------------------------
    run(f"fslmaths {q(args.lr)} -Tmean {q(LR_mean)}", cwd=outdir)
    run(f"bet {q(LR_mean)} {q(LR_mean_brain)} -m", cwd=outdir)
    stdLR = fslstats_scalar(args.lr.resolve(), LR_mask.with_suffix(".nii.gz"))
    print(f"[INFO] LR global std = {stdLR:.6f}")
    run(f"fslmaths {q(args.lr)} -div {stdLR} {q(LR_var1)}", cwd=outdir)

    run(f"fslmaths {q(args.rl)} -Tmean {q(RL_mean)}", cwd=outdir)
    run(f"bet {q(RL_mean)} {q(RL_mean_brain)} -m", cwd=outdir)
    stdRL = fslstats_scalar(args.rl.resolve(), RL_mask.with_suffix(".nii.gz"))
    print(f"[INFO] RL global std = {stdRL:.6f}")
    run(f"fslmaths {q(args.rl)} -div {stdRL} {q(RL_var1)}", cwd=outdir)

    # -----------------------------
    # 2) Concatenate along time
    # -----------------------------
    run(f"fslmerge -t {q(CONCAT)} {q(LR_var1)} {q(RL_var1)}", cwd=outdir)
    # Sanity: TR consistent
    tr_lr = subprocess.check_output(["fslval", str(LR_var1.with_suffix(".nii.gz")), "pixdim4"]).decode().strip()
    tr_rl = subprocess.check_output(["fslval", str(RL_var1.with_suffix(".nii.gz")), "pixdim4"]).decode().strip()
    tr_cc = subprocess.check_output(["fslval", str(CONCAT.with_suffix(".nii.gz")), "pixdim4"]).decode().strip()
    print(f"[CHECK] TRs LR={tr_lr}, RL={tr_rl}, concat={tr_cc}")

    # -----------------------------
    # 3) Motion correction (MCFLIRT)
    # -----------------------------
    run(f"mcflirt -in {q(CONCAT)} -out {q(MC)} -plots -mats -rmsabs -rmsrel", cwd=outdir)

    # -----------------------------
    # 4) Gaussian smoothing (FWHM)
    # -----------------------------
    run(f"fslmaths {q(MC)} -s {sigma:.3f} {q(SMOOTH)}", cwd=outdir)
    print(f"[DONE] Smoothed output: {SMOOTH.with_suffix('.nii.gz')}")
    run(f"bash -lc 'gunzip -c {q(SMOOTH.with_suffix('.nii.gz'))} > {q(SMOOTH.with_suffix('.nii'))}'", cwd=outdir)

    # -----------------------------
    # Optional: FD / outliers
    # -----------------------------
    if args.do_fd:
        conf = outdir/"confounds_fd.txt"; fdplot = outdir/"fd_plot.txt"
        run(f"fsl_motion_outliers -i {q(MC)} -o {q(conf)} --fd --thresh=0.2 -p {q(fdplot)}", cwd=outdir)
        print(f"[INFO] FD confounds -> {conf}")
        print(f"[INFO] FD timecourse -> {fdplot}")

    # -----------------------------
    # 5) Bonus: EPI->T1 coreg (visual only)
    # -----------------------------

    if args.apply_4d_to_t1 :
        if args.t1_brain or args.t1:
            FUNC_MEAN = outdir/"func_mean"
            run(f"fslmaths {q(MC)} -Tmean {q(FUNC_MEAN)}", cwd=outdir)

            if args.t1_brain:
                T1_BRAIN = args.t1_brain.resolve()
                T1_MASK = T1_BRAIN.with_name(T1_BRAIN.stem + "_mask.nii.gz")
                if not T1_MASK.exists():
                    run(f"bet {q(T1_BRAIN)} {q(outdir/'tmp_t1b')} -m", cwd=outdir)
                    T1_MASK = outdir/"tmp_t1b_mask.nii.gz"
            else:
                if args.t1_mask:
                    T1_BRAIN = outdir/"T1_brain"
                    run(f"fslmaths {q(args.t1)} -mas {q(args.t1_mask)} {q(T1_BRAIN)}", cwd=outdir)
                else:
                    T1_BRAIN = outdir/"T1_brain"
                    run(f"bet {q(args.t1)} {q(T1_BRAIN)} -R -f {args.bet_f} -g {args.bet_g} -m", cwd=outdir)
                T1_BRAIN = Path(T1_BRAIN).with_suffix(".nii.gz")

            MAT_EPI2T1 = outdir/"epi2t1.mat"
            FUNC_IN_T1 = outdir/"func_mean_in_T1"
            run(
                f"flirt -in {q(FUNC_MEAN)} -ref {q(T1_BRAIN)} -omat {q(MAT_EPI2T1)} "
                f"-out {q(FUNC_IN_T1)} -dof 6 -cost normmi "
                f"-searchrx -90 90 -searchry -90 90 -searchrz -90 90",
                cwd=outdir,
            )
            print(f"[BONUS] EPI mean in T1 space -> {FUNC_IN_T1.with_suffix('.nii.gz')}")

            MC_IN_T1 = outdir/"subj_concat_var1_mc_in_T1"
            run(f"flirt -in {q(MC)} -ref {q(T1_BRAIN)} -applyxfm -init {q(MAT_EPI2T1)} -out {q(MC_IN_T1)}", cwd=outdir)
            print(f"[BONUS] 4D in T1 space (visual QC) -> {MC_IN_T1.with_suffix('.nii.gz')}")

# --- tiny wrapper to call from Python (keeps CLI unchanged) ---
def run_from_python(
    lr, rl, outdir,
    fwhm=5.0,
    do_fd=False,
    t1_brain=None, t1=None, t1_mask=None,
    bet_f=0.30, bet_g=0.0,
    apply_4d_to_t1=False,
):
    import argparse, sys
    argv = [
        "--lr", str(lr),
        "--rl", str(rl),
        "--outdir", str(outdir),
        "--fwhm", str(fwhm),
    ]
    if do_fd: argv.append("--do_fd")
    if t1_brain: argv += ["--t1_brain", str(t1_brain)]
    if t1:       argv += ["--t1", str(t1)]
    if t1_mask:  argv += ["--t1_mask", str(t1_mask)]
    if apply_4d_to_t1: argv.append("--apply_4d_to_t1")
    argv += ["--bet_f", str(bet_f), "--bet_g", str(bet_g)]
    # Hijack argparse by temporarily replacing sys.argv
    bak = sys.argv
    try:
        sys.argv = ["fsl_preproc.py"] + argv
        main()
    finally:
        sys.argv = bak
