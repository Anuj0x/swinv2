# Swin Transformer V2 Fine-tuning for Flower Classification

This project fine-tunes a Swin Transformer V2 model on a flower dataset using PyTorch and timm.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Ensure your dataset follows this structure:
```
C:\Users\Admin\Downloads\archive\flowers\
├── class1/https://github.com/Anuj0x/swinv2/edit/master/README.md
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## Usage

Simply run the training script:

```bash
python ab.py
```

## Features

- **Model**: Uses `swinv2_base_window8_256` pretrained model from timm
- **GPU Support**: Automatically detects and uses available GPUs
- **Multi-GPU**: Supports multiple GPUs with DataParallel
- **Data Augmentation**: Comprehensive augmentation for better generalization
- **Learning Rate Scheduling**: Cosine annealing for optimal training
- **Monitoring**: Real-time training progress with tqdm
- **Evaluation**: Comprehensive evaluation with confusion matrix and classification report
- **Visualization**: Training curves and confusion matrix plots

## Configuration

You can modify these parameters at the top of `ab.py`:

- `BATCH_SIZE`: Adjust based on GPU memory (default: 32)
- `NUM_EPOCHS`: Number of training epochs (default: 20)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `NUM_WORKERS`: Number of data loading workers (default: 4)

## Output Files

The script generates several output files:

- `best_swinv2_flower_model.pth`: Best model weights
- `training_history.pth`: Training metrics and history
- `confusion_matrix.png`: Confusion matrix visualization
- `training_history.png`: Training and validation curves

## Model Performance

The script will output:
- Training and validation accuracy/loss per epoch
- Final validation accuracy
- Detailed classification report
- Confusion matrix
- Training history plots

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (will use CPU if GPU not available)
- **Memory**: At least 8GB GPU memory for batch size 32
- **Storage**: Sufficient space for dataset and model checkpoints 
