# A Python Toolkit for Machine Learning based Crop Yield Analysis, Prediction, and Benchmarking
This repository trains and evaluates machine learning models for crop yield prediction using the CY-BENCH dataset.

# Setup Instructions
#### Prerequisites
* Python 3.10+
* Conda package manager

#### Installation
##### 1. Create and activate a Conda environment:
```
conda create -n CYP python=3.10
conda activate CYP
```

##### 2.  Clone the CY-BENCH repository:
```
git clone https://github.com/WUR-AI/AgML-CY-BENCH.git
pip install poetry
cd AgML-CY-Bench
poetry install
```

Install the dependencies as indicated at the [CY-BENCH](https://github.com/wur-ai/agml-cy-bench) repository.

##### Step 3: Clone this repository:

```
cd cybench
git clone https://github.com/ambrosia2024/YIELD-HUB.git
mv YIELD-HUB/* ./
rm -rf YIELD-HUB
```

##### Step 4: Install project dependencies:
```
conda env update --name CYP --file environment.yml --prune
```

##### Step 5: Data Download
Download the maize and wheat datasets from [CY-BENCH data](https://zenodo.org/records/13838912) on Zenodo and place them in:

```
AgML-CY-BENCH/cybench/data/
```

The directory structure should look like:
```
AgML-CY-BENCH/
├── cybench/
│   ├── data/
│   │   ├── maize/
│   │   └── wheat/
│   ├── train/           (from YIELD-HUB)
│   ├── process/         (from YIELD-HUB)
│   ├── architectures/   (from YIELD-HUB)
│   ├── wrappers/   (from YIELD-HUB)
|   ├── environment.yml  (from YIELD-HUB
|   └──  (other folders and files from CY-BENCH)
└── (other files from CY-BENCH)
```

# Model training and evaluation

```
cd train/
python statisticalBaselines.py --model mlp --country DE --crop wheat --seed 1111 --save_dir ../output/saved_models/ --output_dir ../output/trained_models/
python tstBaselines.py --crop maize --country NL --model_type tst --use_sota_features --use_residual_trend --use_recursive_lags --use_cwb_feature --aggregation daily --include_spatial_features
python linearBaselines.py --crop maize --country NL --model_type xlinear --use_sota_features --use_residual_trend --use_recursive_lags --use_cwb_feature --aggregation daily --include_spatial_features
```

Alternatively, feel free to use the bash script if you are working with SLURM to train all the baselines together:
```
cd train/
sbatch run_baselines.sh
```

# Model Checkpoints (Currently a private repository)

All the trained checkpoints are available under the [Ambrosia2024/yield-hub](https://huggingface.co/collections/Ambrosia2024/yield-hub) collection on Hugging Face. Individual model repositories include [yield-autoformer-cybench](https://huggingface.co/Ambrosia2024/yield-autoformer-cybench) and the bundled `yield-transformers-cybench` and `yield-linear-cybench` repos referenced by `wrappers/generate_predictions.ipynb`.

# Wrappers
Download these models and place them under the cybench repository under a folder named HFcheckpoints (as shown below) for using the wrappers:
```
AgML-CY-BENCH/
├── cybench/
│   ├── data/
│   │   ├── maize/
│   │   └── wheat/
│   ├── train/           (from YIELD-HUB)
│   ├── HFcheckpoints/           (from YIELD-HUB)
│   ├── process/         (from YIELD-HUB)
│   ├── architectures/   (from YIELD-HUB)
│   ├── wrappers/   (from YIELD-HUB)
|   ├── environment.yml  (from YIELD-HUB
|   └──  (other folders and files from CY-BENCH)
└── (other files from CY-BENCH)
```

# Private Hugging Face Access

For private model repositories, create a token with read access to the `Ambrosia2024` organization repos and store it in `.env`:

```
HF_TOKEN=your_token_here
HUGGINGFACE_HUB_TOKEN=your_token_here
CYBENCH_ROOT=/absolute/path/to/AgML-CY-BENCH
```

The repository includes a scriptable wrapper that downloads `config-and-results.csv` and model checkpoints from Hugging Face using that token:

```
python wrappers/generate_predictions.py --model-type patchtst --country DE --crop maize
```

Predictions are written to `wrappers/data/` by default.

# Packaging

This repository now includes a minimal `pyproject.toml` for editable installs and
wrapper tooling:

```
pip install -e .
yield-hub-predict --model-type patchtst --country DE --crop maize
```

Note: `cybench` still comes from the external `AgML-CY-BENCH` codebase and is not
published on PyPI, so that prerequisite remains separate for now. Point
`CYBENCH_ROOT` at the `AgML-CY-BENCH` repository root.
