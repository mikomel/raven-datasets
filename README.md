# Generalization and Knowledge Transfer in Abstract Visual Reasoning Models

This repository provides implementation of the Attributeless-I-RAVEN and I-RAVEN-Mesh datasets proposed in:
Małkiński, Mikołaj, and Jacek Mańdziuk. "Generalization and Knowledge Transfer in Abstract Visual Reasoning Models." Preprint. Under review. (2024).

Relevant links:
* Main project page: https://github.com/mikomel/raven
* Methods and experiments: https://github.com/mikomel/raven-generalization

Generated datasets are publicly available on [Hugging Face Datasets](https://huggingface.co/docs/datasets/index):
* Attributeless-I-RAVEN: https://huggingface.co/datasets/mikmal/attributeless-i-raven
* I-RAVEN-Mesh: https://huggingface.co/datasets/mikmal/i-raven-mesh

The implementation is forked from:
* https://github.com/husheng12345/SRAN
* https://github.com/Adam-Kowalczyk/I-RAVEN-Mesh

## Setup

The project was implemented in Python 3.9.
Dependencies are listed in `requirements.txt`.

Create a virtual environment and install the required dependencies:
```bash
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Code was formatted with [Black](https://black.readthedocs.io/en/stable/).
It can be installed with:
```bash
pip install black==24.4.2
```

## Usage

Source code is located in the `I-RAVEN/` directory:
```bash
cd I-RAVEN
```

Inspect cli arguments:
```bash
python main.py --help
```

Generate Attributeless-I-RAVEN:
```bash
python main.py --save-dir I-RAVEN-attributeless-color --seed 42 --mesh 0 --color
python main.py --save-dir I-RAVEN-attributeless-position --seed 42 --mesh 0 --position
python main.py --save-dir I-RAVEN-attributeless-size --seed 42 --mesh 0 --size
python main.py --save-dir I-RAVEN-attributeless-type --seed 42 --mesh 0 --type
```

Generate additional regimes (other regimes can be generated analogously):
```bash
# Held-out attribute pairs
python main.py --save-dir I-RAVEN-attributeless-color-size --seed 42 --mesh 0 --color --size
python main.py --save-dir I-RAVEN-attributeless-color-type --seed 42 --mesh 0 --color --type
python main.py --save-dir I-RAVEN-attributeless-size-type --seed 42 --mesh 0 --size --type

# Train set rule different from Constant
python main.py --save-dir I-RAVEN-attributeless-color-progression --seed 42 --mesh 0 --color --color-train-set-rule Progression
python main.py --save-dir I-RAVEN-attributeless-color-arithmetic --seed 42 --mesh 0 --color --color-train-set-rule Arithmetic
python main.py --save-dir I-RAVEN-attributeless-color-distributethree --seed 42 --mesh 0 --color --color-train-set-rule Distribute_Three
```

Generate I-RAVEN-Mesh:
```bash
python main.py --save-dir I-RAVEN-Mesh --seed 42 --mesh 2
```

## Testing

Unit tests can be run with:
```bash
python -m pytest tests
```

## Acknowledgement
This paper builds on the MSc thesis titled "Transfer learning in abstract visual reasoning domain" by Adam Kowalczyk from the Warsaw University of Technology, Warsaw, Poland.
