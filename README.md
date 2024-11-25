# Enhancing Vietnamese Lexical Normalization with Synthetic Data

This repository contains code to evaluate Vietnamese lexical normalization using various synthetic data training strategies and model architectures.

## Project Structure
```
├── config/
│   ├── bartpho.yaml
│   ├── byt5.yaml
│   ├── byt5_dropped.yaml
│   ├── byt5_pre_aug.yaml
│   ├── config.py
│   └── vit5.yaml
├── core/
│   ├── dataset.py
│   ├── executing.py
│   └── modeling.py
├── evaluation/
│   └── err.py
├── logger/
│   └── logger.py
├── README.md
├── requirements.txt
└── run.py
```

## Setup

1. Clone the repository:
    ```
    git clone https://github.com/phong-lt/EnhancingViLexNorm
    ```
2. Install the required packages:
    ```
    pip install -r /EnhancingViLexNorm/requirements.txt
    ```

## Usage

To run the main script:
```bash
python EnhancingViLexNorm/run.py \
	# config file path
	--config-file EnhancingViLexNorm/config/byt5.yaml \
 
	# mode: train - pretrain/train models, eval - evaluate models, predict - predict trained models
	--mode train \

	# evaltype: last - evaluate lattest saved model, best - evaluate best-err saved model 
	--evaltype last \
	
	# predicttype: last - predict lattest saved model, best - predict best-err saved model 
	--predicttype best \
```

## Configuration

The `config/` directory contains YAML files for different model configurations:
- `bartpho.yaml`: Configuration for BARTpho model
- `byt5.yaml`: Configuration for ByT5 model
- `byt5_dropped.yaml`: Configuration for ByT5 model with dropped layers
- `byt5_pre_aug.yaml`: Configuration for ByT5 model with pre-augmentation
- `vit5.yaml`: Configuration for ViT5 model

You can modify these files to adjust model parameters and training settings.

## Core Functionality

- `core/dataset.py`: Handles dataset loading and processing
- `core/executing.py`: Contains execution logic for training and evaluation
- `core/modeling.py`: Defines model architectures and training procedures

## Evaluation

The `evaluation/err.py` file contains the implementation of the Error Reduction Rate (ERR) metric used to evaluate model performance.

## Logging

Logging functionality is implemented in `logger/logger.py`.

---

For any inquiries or issues, please contact the following emails:

- Ms. Thanh-Nhi Nguyen: 21521232@gm.uit.edu.vn
- Mr. Thanh-Phong Le: 21520395@gm.uit.edu.vn

