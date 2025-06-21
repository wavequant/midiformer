# MidiFormer: A Transformer for Symbolic Music Generation

This project implements a decoder-only Transformer model to generate symbolic music in the style of classical piano. The model is built from scratch in PyTorch and trained on the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro), learning the complex patterns and structures of virtuosic piano performances.

The entire project, from data processing to final documentation, is managed through a series of modular Python scripts.

---

## 🎵 Features

- **Advanced MIDI Preprocessing:** A robust pipeline that filters, standardizes, and tokenizes MIDI data.
  - Filters for 4/4 time signature and minimum length.
  - Transposes all pieces to a common key (C Major / A minor) for more effective learning.
  - Converts MIDI into a custom event-based vocabulary (`note_on`, `note_off`, `time_shift`, `velocity`).
- **Decoder-Only Transformer Architecture:** A classic Transformer architecture optimized for autoregressive sequence generation.
- **Efficient Training Harness:**
  - On-the-fly data loading to handle large datasets with minimal memory footprint.
  - Step-based validation on a fixed subset for fast and consistent feedback.
  - Live monitoring and logging of metrics using **Weights & Biases**.
- **Flexible Music Generation:**
  - Supports two distinct generation strategies:
    1.  **Top-K Sampling:** For more creative and varied outputs.
    2.  **Beam Search:** For more coherent and deterministic outputs.
  - Includes **N-gram blocking** to prevent repetitive loops in beam search.
- **Versatile Seeding:** Generation can be seeded with a custom text sequence or with the beginning of a file from the test set for continuation.

---

## 📂 Project Structure

```
.
├── 00_verify_dataset.py       # Verifies the processed data file
├── 01_prepare_dataset.py      # Downloads and processes the MAESTRO dataset
├── 02_train.py                # Trains the Transformer model
├── 03_generate.py             # Generates new music with a trained model
├── config.py                  # All project hyperparameters and settings
├── midi_utils.py              # Helper functions for MIDI processing
├── model.py                   # The MusicTransformer model definition
├── .gitignore                 # Specifies files for Git to ignore
└── README.md                  # This file
```

---

## 🚀 Setup and Usage

### 1. Installation

First, clone the repository, create a virtual environment, and install the required dependencies.

```bash
# Clone the repository
git clone git@github.com:YourUsername/midiformer.git
cd midiformer

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch torchvision torchaudio

# Install project-specific dependencies
pip install mido pretty_midi music21 numpy tqdm wandb python-docx
```

### 2. Data Preparation

1.  Download the **MAESTRO v3.0.0** dataset. You will need the **MIDI files** and the main **`maestro-v3.0.0.csv`** file.
2.  Create a folder named `maestro_midi_data` in the project's root directory.
3.  Place the downloaded MIDI files (e.g., the `2004/`, `2006/`, etc. folders) and the `.csv` file inside `maestro_midi_data`.
4.  Run the preprocessing script. This will take some time as it processes all files.

```bash
python 01_prepare_dataset.py
```
This will create a single file, `data/tokenized_maestro.pkl`, containing all the data needed for training.

### 3. Training the Model

Before starting, make sure you are logged into Weights & Biases for live monitoring.

```bash
# First-time login
wandb login

# Start training
python 02_train.py
```
The script will print a link to your WandB dashboard. Open it to see the training and validation loss in real-time. The best performing model checkpoints will be saved in the `models/` directory, named according to the training step.

### 4. Generating Music

Use the `03_generate.py` script to create new MIDI files using a trained model. Replace `models/best_model_step_XYZ.pt` with the path to your checkpoint.

**Example 1: Creative Generation using Top-K Sampling**
```bash
python 03_generate.py models/best_model_step_XYZ.pt --num_tokens 1500 --top_k 20 --temperature 1.1
```

**Example 2: Coherent Generation using Beam Search**
This uses a beam width of 5 and blocks repeating 3-token sequences to avoid loops.
```bash
python 03_generate.py models/best_model_step_XYZ.pt --num_tokens 1500 --beam_size 5 --block_ngrams 3
```

**Example 3: Continuing a piece from the test set**
This will use the first 100 tokens of the 25th file in the test set as a seed.
```bash
python 03_generate.py models/best_model_step_XYZ.pt --test_seed_index 25 --seed_length 100
```

**Example 4: Comparing two different models with the same seed**
```bash
python 03_generate.py models/model_A.pt models/model_B.pt --test_seed_index 10 --beam_size 5
```
Generated files will be saved in the `generated_music/` directory with descriptive names.
