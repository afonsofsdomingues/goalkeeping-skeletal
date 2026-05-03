# 🥅 Goalkeeping Skeletal Action Recognition

A project for skeletal action recognition in goalkeeping scenarios using advanced pose estimation and machine learning techniques.

> **⚠️ IMPORTANT: Data Availability**: 
> The raw tracking datasets, video footage, and **extracted tensor data** used to train and test these models are **strictly proprietary or derive from proprietary data** and **will not be provided** in this repository or upon request.
> 
> This repository serves purely as a codebase record for the project. To actually execute the pipeline or run the inferences, you will need to supply your own compatible skeletal tracking data (see [Data Configuration](#data-configuration) below).

---

## 📋 Table of Contents

- [Project Documents](#project-documents)
- [Prerequisites](#prerequisites)
- [Project Organization](#project-organization)
- [Environment Setup](#environment-setup)
- [Running the Code](#running-the-code)
- [Model Weights](#model-weights)
- [License](#license)
- [Authors](#authors)

---

## 📚 Project Documents

- [**Final Report**](report.pdf)
- [**Presentation Slides**](presentation.pdf)

---

## ⚙️ Prerequisites

- **Python**: 3.8 or higher (tested with 3.13.9)
- **CUDA** (optional): For GPU acceleration. The code runs on CPU but is optimized for GPU training/inference.
- **Memory**: At least 8GB RAM recommended (16GB+ for comfortable model training).
- **Dependencies**: Listed in `requirements.txt` — includes PyTorch, NumPy, OpenCV, and other ML libraries.

## 📂 Project Organization

The project is structured as follows:

- `animations/`: Scripts for creating skeleton visualizations and extracting match clips.
- `models/`: Contains model definitions and weights.
    - `CrosSCLR/work_dir/crossclr_3views/`: Stores the training weights for the CrosSCLR model.
- `notebooks/`: Jupyter notebooks for analysis and inference.
    - `inference.ipynb`: The primary notebook for running experiments and generating results.
    - `exploration.ipynb`, `ntu_inference.ipynb`, etc.: Used for initial data exploration and testing.
- `preprocessing/`: Scripts for the data processing pipeline (ETL).
    - `run_pipeline.py`: The main script to process raw data into tensors.
- `utils/`: Utility functions used across the project.

## 🛠️ Environment Setup

To set up the development environment, execute the following commands in your terminal:

1. **Create a virtual environment:**
    ```bash
    python -m venv .venv
    ```

2. **Activate the virtual environment:**
    - On **Windows**:
      ```powershell
      .venv\Scripts\activate
      ```
    - On **macOS/Linux**:
      ```bash
      source .venv/bin/activate
      ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Running the Code

### Data Configuration

To run the full data pipeline and process raw tracking data into model-ready tensors:

1. **Raw Data**: Place data in `data/euro2024/`, with one folder per match.
    - The pipeline assumes a `.7z` file containing tracking JSONs exists within the match folder.
    - Video extraction scripts depend on the match footage file being present there as well.

2. **Processing Data**:
    ```bash
    python preprocessing/run_pipeline.py
    ```
    This script executes three steps:
    1. **Filtering**: Extracts and filters raw tracking data (`preprocessing/filter_data.py`).
    2. **NTU Tensor Generation**: Creates standard tensors in `data/tensor_ntu/`.
    3. **Native Tensor Generation**: Creates native resolution tensors in `data/tensor_native/`.

### Inference and Analysis

- **Main Analysis**: Open `notebooks/inference.ipynb` to run the main experiments. This notebook loads the processed tensors and model weights to perform action recognition and analysis.
- **Exploration**: Other notebooks in the `notebooks/` directory contain experimental code and exploratory data analysis.

## 🧠 Model Weights

The CrosSCLR model weights are located in `models/CrosSCLR/work_dir/crossclr_3views/`. These weights are required for inference in `notebooks/inference.ipynb`.

## 📄 License

This project is provided as-is for research and educational purposes. Please refer to the LICENSE file for specific terms. Note that the proprietary nature of the datasets restricts redistribution.

## ✍️ Authors

- Afonso Domingues
- Kaushik Karthikeyan
