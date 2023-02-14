from pathlib import Path
from setuptools import find_packages, setup

import torch

PROJECT_NAME = "3DJCG"
PACKAGE_NAME = PROJECT_NAME.replace("-", "_")
DESCRIPTION = "Joint Captioning and Grounding"

TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 13], "Requires PyTorch >= 1.13"


if __name__ == "__main__":
    version = "0.1.0"

    print(f"Building {PROJECT_NAME}-{version}")

    setup(
        name=PROJECT_NAME,
        version=version,
        author="Rui Zheng",
        author_email="1156486807@qq.com",
        url=f"https://github.com/yjzhao19981027/{PROJECT_NAME}",
        download_url=f"https://github.com/yjzhao19981027/{PROJECT_NAME}/tags",
        description=DESCRIPTION,
        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",
        packages=find_packages(exclude=("tests",)),
        package_data={PACKAGE_NAME: ["*.dll", "*.so", "*.dylib", "*.txt", "*.txt.gz"]},
        zip_safe=False,
        python_requires=">=3.9",
        install_requires=[
            "pillow",
            "aiofiles",
            "fastapi",
            "uvicorn[standard]",
            "python-multipart",
            "ftfy",
            "regex",
            "tqdm",
        ],
    )
