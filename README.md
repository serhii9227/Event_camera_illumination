# Event Camera Illumination

This repository contains the code accompanying the diploma project **"Improving Illumination Change Robustness for Event-Based Vision Models in Single Object Tracking"**.

## Overview

The project investigates how event-based single object trackers can be made more robust to abrupt illumination changes (flashes). Two state-of-the-art trackers, **SDTrack** and **HDETrack**, are extended with a modular pipeline consisting of:

- Fine-tuning on synthetically augmented sequences
- A Kalman filter for trajectory stabilization during the first frames of the flash
- A re-detection module activated after the flash

## Repository structure

- `HDETrack/` — HDETrack tracker scrypts needed to apply changes in pipeline 
- `SDTrack/` — SDTrack tracker scrypts needed to apply changes in pipeline 
- `GTP_FE108_csv.py` — GTP-frame generator from event CSV files
- `create_events.ipynb` — Generation of synthetic events from RGB video (V2E)
- `SDTrack.ipynb` — SDTrack pipeline: GTP frame creation, data augmentation, fine-tuning, re-detection, Kalman
- `HDETrack.ipynb` — HDETrack pipeline: fine-tuning, re-detection, Kalman

## Notebooks

### `create_events.ipynb`
Demonstrates how synthetic events are generated from RGB video using the V2E pipeline. The output consists of DVS-style event frames and a CSV file containing the raw event stream.

### `SDTrack.ipynb`
Demonstrates the full SDTrack pipeline, including:
- Construction of GTP (Global Trajectory Prompt) frames
- Generation of augmented sequences for fine-tuning
- Fine-tuning of the tracker on flash-affected data
- Inference with re-detection and Kalman filtering

### `HDETrack.ipynb`
Demonstrates the HDETrack pipeline with fine-tuning, re-detection, and Kalman filtering. HDETrack operates directly on signed DVS frames, so GTP construction is not needed here.

## Resources

The notebooks were developed in Google Colab with data stored on Google Drive. To reproduce the experiments, the following resources are required:

| Resource | Link |
|---|---|
| Pre-trained and fine-tuned checkpoints | [Google Drive](https://drive.google.com/drive/u/0/folders/1SvPpeXBZmEkb4T4opgZhOw79uoxfH-uZ) |
| Test benchmark (39 sequences across three flicker frequencies) | [Google Drive](https://drive.google.com/drive/u/0/folders/1At3VR3Sb2aNJZHg2PWVEurj20cKqIExb) |
| Augmented training sequences (Augmentation A, Augmentation B, originals) | [Google Drive](https://drive.google.com/drive/u/0/folders/1lbcwgcgDrCCnWrZnHG2zDispDy1dJXG-) |

## Setup notes

The notebooks expect data to be mounted from Google Drive. If you run them in a different environment, please update the data paths at the top of each notebook to match your local setup.
