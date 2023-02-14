from pathlib import Path
from setuptools import find_packages, setup

import torch

PROJECT_NAME = "ThreeDJCG"
PACKAGE_NAME = PROJECT_NAME.replace("-", "_")
DESCRIPTION = "Joint Captioning and Grounding"

TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 7], "Requires PyTorch >= 1.7"


if __name__ == "__main__":
    version = "0.1.12"

    print(f"Building {PROJECT_NAME}-{version}")

    setup(
        name=PROJECT_NAME,
        version=version,
        author="Rui Zheng",
        author_email="1156486807@qq.com",
        url=f"https://github.com/ScaryJellyfish/{PROJECT_NAME}",
        download_url=f"https://github.com/ScaryJellyfish/{PROJECT_NAME}/tags",
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=("tests",)),
        package_data={PACKAGE_NAME: ["./ThreeDJCG/data/*.json", "./ThreeDJCG/data/*.p", "./ThreeDJCG/data/scannet/*.npz", "./ThreeDJCG/data/scannet/*.txt", "./ThreeDJCG/data/scannet/*.tsv"]},
        zip_safe=False,
        python_requires=">=3.7",
        install_requires=[
            "pillow",
            "aiofiles",
            "fastapi",
            "uvicorn[standard]",
            "python-multipart",
            "ftfy",
            "regex",
            "tqdm",
            "h5py",
            "easydict"
        ],
    )
