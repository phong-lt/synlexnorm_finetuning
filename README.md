# Enhancing Vietnamese Lexical Normalization with Synthetic Data

## Setup

1. Clone the repository:
    ```
    git clone https://github.com/phong-lt/synlexnorm_finetuning
    ```
2. Install the required packages:
    ```
    pip install -r /synlexnorm_finetuning/requirements.txt
    ```

## Usage

To run the main script:
```bash
python synlexnorm_finetuning/run.py \
	# config file path
	--config-file synlexnorm_finetuning/config/byt5.yaml \
 
	# mode: train - pretrain/train models, eval - evaluate models, predict - predict trained models
	--mode train \

	# evaltype: last - evaluate lattest saved model, best - evaluate best-err saved model 
	--evaltype last \
	
	# predicttype: last - predict lattest saved model, best - predict best-err saved model 
	--predicttype best \
```