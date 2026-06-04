#!/usr/bin/env python3
"""
install_packages.py — Install all pipeline dependencies.

Usage:
    python install_packages.py           # install everything
    python install_packages.py --check   # check only, no install
    python install_packages.py --quiet   # suppress pip output
"""

import subprocess, sys, importlib, argparse

PACKAGES = [
    ("numpy",       "numpy>=1.24",       True),
    ("pandas",      "pandas>=1.5",       True),
    ("scipy",       "scipy>=1.10",       True),
    ("sklearn",     "scikit-learn>=1.3", True),
    ("matplotlib",  "matplotlib>=3.7",   True),
    ("seaborn",     "seaborn>=0.12",     True),
    ("joblib",      "joblib>=1.2",       True),
    ("statsmodels", "statsmodels>=0.14", True),
    ("GEOparse",    "GEOparse>=2.0",     True),
    ("shap",        "shap>=0.43",        True),
    ("imblearn",    "imbalanced-learn",  False),
    ("adjustText",  "adjustText",        False),
]

G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[1m"; E = "\033[0m"

def check(mod):
    try:
        m = importlib.import_module(mod)
        return True, getattr(m, "__version__", "ok")
    except ImportError:
        return False, "missing"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check",  action="store_true")
    ap.add_argument("--quiet",  action="store_true")
    args = ap.parse_args()

    print(f"\n{B}T2D ML Pipeline — Package Installer{E}")
    print(f"Python: {sys.version.split()[0]}\n")
    print(f"  {'Package':<22} {'Status':<14} Version")
    print(f"  {'-'*50}")

    to_install = []
    for mod, spec, req in PACKAGES:
        ok, ver = check(mod)
        tag = "[required]" if req else "[optional]"
        if ok:
            print(f"  {G}✓{E}  {mod:<20} {G}{'installed':<14}{E} {ver}")
        else:
            c = R if req else Y
            print(f"  {c}✗{E}  {mod:<20} {c}{'missing':<14}{E} {tag}")
            to_install.append((mod, spec, req))

    print()
    if not to_install:
        print(f"  {G}{B}All packages present.{E}\n")
        return
    if args.check:
        print(f"  {Y}{len(to_install)} package(s) would be installed.{E}\n")
        return

    for mod, spec, req in to_install:
        print(f"  Installing {spec} ...", end=" ", flush=True)
        cmd = [sys.executable, "-m", "pip", "install", spec, "--break-system-packages"]
        if args.quiet:
            cmd.append("-q")
        r = subprocess.run(cmd, capture_output=args.quiet)
        ok, ver = check(mod)
        if ok:
            print(f"{G}✓ {ver}{E}")
        else:
            print(f"{R}FAILED{E}" if req else f"{Y}skipped{E}")

    print(f"\n  {G}Done.{E}\n")

if __name__ == "__main__":
    main()
