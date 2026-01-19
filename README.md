# Audio-Based Navigation using k-Wave

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![k-Wave](https://img.shields.io/badge/k--Wave-0.3.0+-green.svg)](https://github.com/waltsims/k-wave-python)

**PhD-Research Quality** implementation for training autonomous agents to navigate 2D mazes using **only acoustic reverberations**. This project uses the **k-Wave library** for physically accurate acoustic simulations and deep learning for navigation policy learning.

---

## Overview

This repository demonstrates how an agent can learn to navigate complex environments using sound-based perception. The approach combines:

- **k-Wave acoustic simulations** for physically accurate sound propagation
- **Multi-channel microphone arrays** for directional acoustic perception
- **Deep Convolutional Neural Networks** for learning navigation policies
- **A\* pathfinding oracle** for supervised learning labels

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/djpitzele/Audio-Maze-Navigation.git
cd Audio-Maze-Navigation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Training Data

Open and run `notebooks/01_Data_Creation.ipynb` to:
- Generate a maze environment
- Run k-Wave acoustic simulations for all walkable positions
- Create labeled dataset with (spectrogram, action) pairs
- Save to HDF5 format

**Warning:** Data generation is computationally intensive!
- 20×20 maze (~250 positions): **2-4 hours**
- 50×50 maze (~1500 positions): **12-25 hours**

Use small mazes (10×10 or 15×15) for initial testing.

### 3. Train Navigation Model

Open and run `notebooks/02_Model_Training.ipynb` to:
- Load pre-computed acoustic dataset
- Train CNN to predict actions from spectrograms
- Evaluate performance and save model

Typical training time: **10-30 minutes** (depends on dataset size and GPU availability)

### 4. Run Interactive Demo

Open and run `notebooks/03_Simulation_Demo.ipynb` to:
- Load trained model
- Generate test maze
- Watch agent navigate using acoustic perception
- Compare with optimal A* path

---

## 📁 Repository Structure

```
Audio-Maze-Navigation/
├── data/                          # Generated datasets (empty initially)
│   ├── acoustic_navigation_data.h5
│   └── audio_nav_model.pth
├── notebooks/                     # Interactive Jupyter notebooks
│   ├── 01_Data_Creation.ipynb    # k-Wave simulation & data generation
│   ├── 02_Model_Training.ipynb   # CNN training pipeline
│   └── 03_Simulation_Demo.ipynb  # Interactive navigation demo
├── src/                           # Core Python modules
│   ├── __init__.py
│   ├── simulation.py              # k-Wave acoustic simulator
│   ├── environment.py             # Maze generation & A* oracle
│   ├── model.py                   # CNN architectures
│   ├── dataset.py                 # PyTorch data loaders
│   └── utils.py                   # Visualization utilities
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

For a paper-style writeup of the project, see [here](https://djpitzele.github.io/files/CMSC818V_Final_Project_Report.pdf).

*Note: AI assistance was used for portions of the code and for this README.*
