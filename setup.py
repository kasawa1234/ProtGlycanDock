import sys
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
# Read requirements from the file
with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

# Check if the user specified the CPU option
if "--cpu" in sys.argv:
    # Remove the gpu packages
    try:
        to_drop = [x for x in install_requires if "nvidia" in x or "cuda" in x]
        for x in to_drop:
            install_requires.remove(x)
    except ValueError:
        pass
    # Remove the --cpu option from sys.argv so setuptools doesn't get confused
    sys.argv.remove("--cpu")

setup(
    name="protenix",
    python_requires=">=3.11",
    version="0.6.2",
    description="ProtGlycanDock: Protenix-based protein-glycan docking model and benchmark code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Minghao Xu, Jiaze Song, Jinyu Wang, Wentao Zhang",
    url="https://github.com/kasawa1234/ProtGlycanDock",
    project_urls={
        "Dataset": "https://huggingface.co/datasets/ProtGlycanDock/ProtGlycanDock",
        "Issues": "https://github.com/kasawa1234/ProtGlycanDock/issues",
    },
    packages=find_packages(
        exclude=(
            "assets",
            "benchmark",
            "*.egg-info",
        )
    ),
    include_package_data=True,
    package_data={
        "protenix": ["model/layer_norm/kernel/*"],
    },
    install_requires=install_requires,
    license="Apache 2.0 License",
    platforms="manylinux1",
    entry_points={
        "console_scripts": [
            "protenix = runner.batch_inference:protenix_cli",
        ],
    },
)
