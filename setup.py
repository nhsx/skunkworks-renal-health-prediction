# coding=utf-8
# Copyright (c) 2022 NHS England
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# This project incorporates work covered by the following copyright and permission notice:
#
#     Copyright 2021 Google Health Research.
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#             http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""Script for installing package with setuptools."""
from setuptools import find_packages, setup

import aki_predictions


with open("README.md", encoding="utf8") as f:
    long_description = f.read()

with open("LICENCE", encoding="utf8") as f:
    licence = f.read()


tests_requires = [
    "flake8==3.9.2",
    "flake8-import-order==0.18.1",
    "flake8-import-style==0.5.0",
    "flake8-docstrings",
    "pytest==7.1.2",
    "pytest-cov==3.0.0",
    "pytest-assume==2.4.3",
    "mock==4.0.3",
    "pre-commit==2.19.0",
    "XlsxWriter==3.0.3",
]


experiments_requires = [
    "jupyter==1.0.0",
]


docs_requires = [
    "sphinx==5.1.0",
    "sphinxcontrib-napoleon==0.7",
    "mistune~=0.8.4",
    "m2r2==0.3.2",
    "sphinx-markdown-builder==0.5.5",
]


cpu_requires = ["tensorflow==1.15.0"]

gpu_requires = ["tensorflow-gpu==1.15.0"]

install_requires = [
    "setuptools",
    "absl-py==0.8.1",
    "dm-sonnet==1.36",
    "dm-tree==0.1.1",
    "numpy==1.18.1",
    "tensorboard==1.15.0",
    "tensorflow-estimator==1.15.1",
    "tensorflow-hub==0.12.0",
    "tensorflow-probability==0.8.0",
    "tf-slim==1.1.0",
    "pandas==1.3.5",
    "openpyxl==3.0.10",
    "jsonschema==4.6.0",
    "tqdm==4.64.0",
    "jsonlines==3.0.0",
    "protobuf==3.20.1",
    "matplotlib==3.5.2",
]


setup(
    name="aki_predictions",
    description="C308 NHS AI Predictive Renal Health PoC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Google LLC, Roke Manor Research Ltd",
    licence=licence,
    version=aki_predictions.__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "cpu": cpu_requires,
        "gpu": gpu_requires,
        "tests": tests_requires,
        "experiments": experiments_requires,
        "docs": docs_requires,
    },
    python_requires="~=3.7.5",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "aki_predictions_tool = aki_predictions.__main__:main",
        ],
    },
)
