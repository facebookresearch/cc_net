# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

setup(
    name="cc_net",
    version="0.1.0",
    packages=["cc_net"],
    # metadata to display on PyPI
    author="Guillaume Wenzek",
    author_email="guw@fb.com",
    description="Tools to download and clean Common Crawl",
    keywords="common crawl dataset",
    url="https://github.com/facebookresearch/cc_net",
    project_urls={
        "Bug Tracker": "https://github.com/facebookresearch/cc_net/issues",
        "Source Code": "https://github.com/facebookresearch/cc_net",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">=3.7",
    install_requires=[
        "beautifulsoup4>=4.7.1",
        "pandas>=0.23.4",
        "requests>=2.22.0",
        "fasttext>=0.9.1",
        "sentencepiece>=0.1.82",
        "kenlm @ https://github.com/kpu/kenlm/archive/master.zip",
        "func_argparse>=1.0.3",
        "sacremoses",
    ],
    extras_require={
        "dev": ["mypy>=0.730", "pytest", "black", "isort"],
        # To use scripts inside cc_net/tools
        "tools": ["sentence_splitter"],
        # Allows to run on a SLURM cluster. Not open sourced yet.
        "slurm": ["submitit"],
        # Memory-efficient hashset.
        # This fork only compiles the kind of dict used by cc_net.
        # Full version is at https://github.com/atom-moyer/getpy
        "getpy": ["getpy @ git+https://github.com/gwenzek/getpy.git@v0.9.9-subset"],
    },
)
