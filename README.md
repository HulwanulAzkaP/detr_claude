# DETR Custom Implementation

This repository contains an implementation of DETR (DEtection TRansformer) for training on custom datasets using the COCO JSON format from Roboflow.

## Project Structure

```
detr_custom/
│
├── config/          # Configuration files
├── data/           # Dataset and dataloader implementations
├── dataset/        # Your custom dataset files
├── models/         # Model implementations
├── utils/          # Utility functions
├── output/         # Output directory for models and predictions
├── requirements.txt
├── main.py
└── README.md
```

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place your Roboflow COCO format dataset in the `dataset` folder
   - Ensure the following structure:
     ```
     dataset/
     ├── train/
     │   ├── _annotations.coco.json
     │   └── [training images]
     ├── valid/
     │   ├── _annotations.coco.json
     │   └── [validation images]
     └── test/
         ├── _annotations.coco.json
         └── [test images]
     ```

## Usage

The implementation supports four modes of operation:

### Training

To train the model:
```bash
python main.py --mode train
```

### Evaluation

To evaluate the model on the validation set:
```bash
python main.py --mode eval --model_path path/to/model.pth
```

### Inference

To run inference on the test set:
```bash
python main.py --mode inference --model_path path/to/model.pth
```

### Save Model

To save the current model state:
```bash
python main.py --mode save --model_path output/saved_models/my_model.pth
```

## Configuration

You can modify the model and training parameters in `config/config.py`. Key parameters include:

- Model architecture parameters (hidden dimensions, number of heads, etc.)
- Training parameters (batch size, learning rate, etc.)
- Dataset parameters (image size, number of workers, etc.)

## Output

The implementation automatically creates the following output directories:

- `output/saved_models/`: Saved model checkpoints
- `output/predictions/`: Inference results (CSV and COCO JSON format)
- `output/metrics/`: Evaluation metrics

## Model Details

This implementation uses:
- ResNet50 backbone
- Transformer encoder-decoder architecture
- Hungarian matching for loss computation
- AdamW optimizer with learning rate scheduling

## Reference

This implementation is based on the original DETR paper:
"End-to-End Object Detection with Transformers" (Carion et al., 2020)