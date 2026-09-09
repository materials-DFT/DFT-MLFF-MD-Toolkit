#!/usr/bin/env python3
"""Convert ALLEGRO .ckpt checkpoints to LAMMPS-compatible .nequip.pt2/.nequip.pth.

Submits `nequip-compile` to a GPU compute node.  Accepts either a checkpoint file
or a directory to search for one.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

NEQUIP_PYTHON = "/Users/924322630/miniconda3/envs/nequip/bin/python"

# Toolchains that work on this cluster, newest first.  /usr/bin/g++ is 8.5, too
# old for the AOTInductor C++17 codegen, and the conda base toolchain
# (x86_64-conda-linux-gnu-gcc + its sysroot) is broken -- any #include <stdlib.h>
# fails with "conflicting types for 'timespec_get'".
GCC_TOOLSETS = [
    "/opt/rh/gcc-toolset-13/root/usr",
    "/opt/rh/gcc-toolset-12/root/usr",
    "/opt/rh/gcc-toolset-11/root/usr",
]

# Compiler env vars conda's base environment exports.  sbatch propagates the
# submitting shell's environment, so torch/triton would otherwise pick up the
# broken conda toolchain on the compute node.
CONDA_COMPILER_VARS = [
    "CC", "CXX", "CPP", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
    "DEBUG_CFLAGS", "DEBUG_CXXFLAGS", "CC_FOR_BUILD", "CXX_FOR_BUILD",
    "CONDA_BUILD_SYSROOT", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH", "GCC", "GCC_AR", "GCC_NM", "GCC_RANLIB", "HOST", "BUILD",
    "host_alias", "build_alias", "AR", "AS", "LD", "NM", "RANLIB", "STRIP",
    "OBJCOPY", "OBJDUMP", "READELF", "ADDR2LINE", "ELFEDIT", "GPROF", "SIZE",
    "STRINGS", "CXXFILT", "LD_GOLD",
]

CUDA_HOME_SHIM = os.path.expanduser("~/.local/cuda_home_cu118")


def find_checkpoint(path):
    """Resolve a user-supplied path to a single .ckpt file.

    A file is used as-is.  A directory is searched (itself, then models/, then
    one level down) preferring best.ckpt, then last.ckpt, then a lone .ckpt.
    """
    if os.path.isfile(path):
        return path

    if not os.path.isdir(path):
        print(f"❌ Not found: {path}")
        sys.exit(1)

    # Tiers, most specific first: the directory itself, its models/ subdirectory,
    # then one level down.  Only the first tier with any hits is considered.
    tiers = [
        sorted(glob.glob(os.path.join(path, "*.ckpt"))),
        sorted(glob.glob(os.path.join(path, "models", "*.ckpt"))),
        sorted(glob.glob(os.path.join(path, "*", "*.ckpt"))),
    ]
    found = next((t for t in tiers if t), [])

    if not found:
        print(f"❌ No .ckpt file found in {path} (looked in ./, models/, */)")
        sys.exit(1)

    # best.ckpt/last.ckpt only disambiguates within one directory -- picking one
    # run out of several sibling directories is never a safe guess.
    dirs = {os.path.dirname(f) for f in found}
    if len(dirs) == 1:
        for preferred in ("best.ckpt", "last.ckpt"):
            for f in found:
                if os.path.basename(f) == preferred:
                    if len(found) > 1:
                        print(f"🔎 Found {len(found)} checkpoints in "
                              f"{os.path.dirname(f)}, using {preferred}")
                    return f
        if len(found) == 1:
            return found[0]

    print(f"❌ Ambiguous: {len(found)} checkpoints under {path}, "
          f"none obviously the one to convert:")
    for f in found[:10]:
        print(f"     {f}")
    if len(found) > 10:
        print(f"     ... and {len(found) - 10} more")
    print("   Re-run naming the checkpoint (or its directory) explicitly.")
    sys.exit(1)


def ensure_cuda_home_shim():
    """Build a CUDA_HOME that torch's C++ codegen can actually use.

    torch infers CUDA_HOME from `which nvcc`, which on this cluster is NVHPC's.
    That puts .../nvidia_hpc_sdk/.../compilers/include on the compile line, whose
    clang-style intrinsics headers make g++ die on immintrin.h.  Instead point
    CUDA_HOME at the cu11 headers shipped with torch itself.  The pip wheel omits
    the crt/ subdirectory, so borrow that from any real toolkit on the box.

    Returns the shim path, or None if it could not be built (in which case the
    job just falls back to whatever torch infers).
    """
    site_pkgs = glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(NEQUIP_PYTHON)),
        "lib", "python*", "site-packages", "nvidia", "cuda_runtime"))
    if not site_pkgs:
        return None
    src_inc = os.path.join(site_pkgs[0], "include")
    src_lib = os.path.join(site_pkgs[0], "lib")
    if not os.path.isdir(src_inc):
        return None

    crt = None
    for cand in sorted(glob.glob("/Users/924322630/nvidia_hpc_sdk/install/*/*/cuda/include/crt")) + \
                sorted(glob.glob("/usr/local/cuda*/include/crt")) + \
                [os.path.join(src_inc, "crt")]:
        if os.path.isfile(os.path.join(cand, "host_defines.h")):
            crt = cand
            break
    if crt is None:
        return None

    inc = os.path.join(CUDA_HOME_SHIM, "include")
    lib = os.path.join(CUDA_HOME_SHIM, "lib64")
    os.makedirs(inc, exist_ok=True)
    os.makedirs(lib, exist_ok=True)

    for entry in os.listdir(src_inc):
        if entry == "__pycache__":
            continue
        dst = os.path.join(inc, entry)
        if os.path.islink(dst) or os.path.isfile(dst):
            os.remove(dst)
        if not os.path.exists(dst):
            os.symlink(os.path.join(src_inc, entry), dst)

    # crt/ last, so it overrides the flattened copies in the wheel
    dst_crt = os.path.join(inc, "crt")
    if os.path.islink(dst_crt):
        os.remove(dst_crt)
    if not os.path.exists(dst_crt):
        os.symlink(crt, dst_crt)

    for so in glob.glob(os.path.join(src_lib, "libcudart.so*")):
        for name in ("libcudart.so", os.path.basename(so)):
            dst = os.path.join(lib, name)
            if os.path.islink(dst):
                os.remove(dst)
            if not os.path.exists(dst):
                os.symlink(so, dst)

    return CUDA_HOME_SHIM


def build_job_script(model_path, output_path, device, mode, cuda_home):
    toolset = next((t for t in GCC_TOOLSETS if os.path.isdir(t)), None)
    model_dir = os.path.dirname(model_path) or os.getcwd()

    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=nqcompile",
        "#SBATCH --partition=gpucluster",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --nodes=1",
        "#SBATCH --time=00:30:00",
        "set -uo pipefail",
        "",
        "# Strip conda's compiler env vars: sbatch propagates the submitting",
        "# shell's environment, and the conda base toolchain is broken here",
        "# (#include <stdlib.h> -> conflicting types for 'timespec_get').",
        "unset " + " ".join(CONDA_COMPILER_VARS) + " || true",
        "",
    ]

    if toolset:
        lines += [
            "# /usr/bin/g++ is too old for the AOTInductor C++17 codegen.",
            f"export PATH={toolset}/bin:$PATH",
            f"export LD_LIBRARY_PATH={toolset}/lib64:${{LD_LIBRARY_PATH:-}}",
            f"export CC={toolset}/bin/gcc",
            f"export CXX={toolset}/bin/g++",
            "",
        ]

    lines += [
        "# torch infers CUDA_HOME from `which nvcc`, which is NVHPC's here; its",
        "# include/ holds clang-style intrinsics that break g++ on immintrin.h.",
        "export PATH=$(printf '%s' \"$PATH\" | tr ':' '\\n' | grep -v nvidia_hpc_sdk | paste -sd:) || true",
    ]
    if cuda_home:
        lines += [
            f"export CUDA_HOME={cuda_home}",
            "export CUDA_PATH=$CUDA_HOME",
        ]
    lines += [
        "",
        "export TORCHINDUCTOR_COMPILE_THREADS=1",
        "",
        "# Checkpoints record their training dataset by relative path, so run",
        "# from the checkpoint's directory.",
        f'cd "{model_dir}"',
        "",
        f'"{NEQUIP_PYTHON}" -m nequip.scripts.compile \\',
        f'    "{model_path}" \\',
        f'    "{output_path}" \\',
        f"    --device {device} \\",
    ]
    if mode == "aotinductor":
        lines.append("    --mode aotinductor --target pair_allegro")
    else:
        lines.append("    --mode torchscript")
    lines += [
        "rc=$?",
        'if [ "$rc" -eq 0 ]; then',
        f'    echo "✅ Compiled: {output_path}"',
        "else",
        '    echo "❌ nequip-compile failed (exit $rc)"',
        "fi",
        "exit $rc",
        "",
    ]
    return "\n".join(lines)


def run_on_compute_node(model_path, output_name=None, device="cuda",
                        mode="aotinductor", keep_logs=False):
    """Submit the conversion to a GPU compute node."""
    model_path = os.path.abspath(model_path)
    model_basename = os.path.basename(model_path)
    workdir = os.getcwd()

    base_name = output_name or os.path.splitext(model_basename)[0]
    ext = "nequip.pt2" if mode == "aotinductor" else "nequip.pth"
    output_path = os.path.join(workdir, f"{base_name}.{ext}")

    cuda_home = ensure_cuda_home_shim()
    script_content = build_job_script(model_path, output_path, device, mode, cuda_home)

    # Written outside the working directory; sbatch reads it at submit time.
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(script_content)
        script_name = fh.name
    os.chmod(script_name, 0o755)

    if keep_logs:
        out_path = os.path.join(workdir, f"convert_{base_name}.out")
        err_path = os.path.join(workdir, f"convert_{base_name}.err")
    else:
        out_path = err_path = "/dev/null"

    print("🚀 Submitting conversion job to compute node...")
    result = subprocess.run([
        "sbatch",
        "-p", "gpucluster",
        "--gres", "gpu:1",
        "--chdir", workdir,
        "--output", out_path,
        "--error", err_path,
        script_name,
    ], capture_output=True, text=True)

    try:
        os.remove(script_name)
    except OSError:
        pass

    if result.returncode != 0:
        print(f"❌ Failed to submit job: {result.stderr}")
        return None

    job_id = result.stdout.strip().split()[-1]
    print(f"✅ Job submitted successfully! Job ID: {job_id}")
    print(f"📤 Output file: {output_path}")
    if keep_logs:
        print(f"📋 Monitor progress: tail -f {out_path}")
        print(f"🔍 Check errors: tail -f {err_path}")
    else:
        print("📋 Logs are discarded; re-run with --keep-logs to inspect a failure")
    print("⏱️  Job typically takes 1-2 minutes to complete")
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description="Convert ALLEGRO .ckpt files to LAMMPS-compatible .nequip.pt2/.nequip.pth format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (AOTInductor mode, default)
  python convert_allegro_to_lammps.py best.ckpt

  # Point at a directory -- best.ckpt / last.ckpt is found automatically,
  # in the directory itself or in its models/ subdirectory
  python convert_allegro_to_lammps.py .

  # Specify output name
  python convert_allegro_to_lammps.py best.ckpt --output best_model

  # Use TorchScript mode
  python convert_allegro_to_lammps.py best.ckpt --mode torchscript

  # Keep the Slurm .out/.err files (discarded by default)
  python convert_allegro_to_lammps.py . --keep-logs
        """,
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=".",
        help="ALLEGRO .ckpt checkpoint, or a directory to search for one (default: .)",
    )
    parser.add_argument(
        "--output", "-o", dest="output_name",
        help="Output filename (without extension). Default: same as checkpoint filename",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda",
        help="Device to use for compilation (default: cuda)",
    )
    parser.add_argument(
        "--mode", choices=["torchscript", "aotinductor"], default="aotinductor",
        help="Compilation mode: 'aotinductor' (.nequip.pt2, default, requires PyTorch >= 2.6) "
             "or 'torchscript' (.nequip.pth, works with PyTorch 2.4+)",
    )
    parser.add_argument(
        "--keep-logs", action="store_true",
        help="Write convert_<name>.out/.err instead of discarding job output",
    )

    args = parser.parse_args()

    model_path = find_checkpoint(args.model_path)
    if not model_path.endswith(".ckpt"):
        print(f"⚠ Warning: {model_path} doesn't end with .ckpt (may not be an ALLEGRO checkpoint)")

    print(f"🖥️  Submitting {model_path} to compute node...")
    print(f"   Device: {args.device}")
    print(f"   Mode: {args.mode}")

    job_id = run_on_compute_node(
        model_path,
        output_name=args.output_name,
        device=args.device,
        mode=args.mode,
        keep_logs=args.keep_logs,
    )
    if job_id:
        print(f"✅ Conversion job submitted for {model_path}")


if __name__ == "__main__":
    main()
