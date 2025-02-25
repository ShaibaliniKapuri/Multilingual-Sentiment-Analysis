# Multi-Lingual Sentiment Analysis with LLaMA

## Overview

This notebook demonstrates the process of performing multi-lingual sentiment analysis using the LLaMA (Large Language Model Meta AI) model. The goal is to classify text sentences in various Indian languages as either "positive" or "negative" sentiment. The notebook leverages the Hugging Face Transformers library, along with the PEFT (Parameter-Efficient Fine-Tuning) and LoRA (Low-Rank Adaptation) techniques to fine-tune the LLaMA model efficiently.

## Key Features

- **Multi-Lingual Support**: The model is trained to analyze sentiment in multiple Indian languages, including Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, and Urdu.
- **Efficient Fine-Tuning**: Utilizes LoRA and PEFT for memory-efficient fine-tuning of the LLaMA model.
- **Sentiment Classification**: The model predicts whether a given sentence has a positive or negative sentiment.
- **Evaluation**: Includes functions to evaluate the model's performance using metrics like F1 score, classification report, and confusion matrix.

## Requirements

To run this notebook, you need the following Python packages installed:

- `transformers`
- `bitsandbytes`
- `accelerate`
- `peft`
- `datasets`
- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `trl`
- `wandb`

You can install these packages using the following command:

```bash
!pip install -q bitsandbytes accelerate peft transformers datasets torch pandas numpy scikit-learn matplotlib trl wandb
```

## Dataset

The dataset used in this notebook is a multi-lingual sentiment analysis dataset containing sentences in various Indian languages, each labeled as either "positive" or "negative". The dataset is split into training and validation sets for model training and evaluation.

### Dataset Structure

- **train.csv**: Contains the training data with columns `ID`, `language`, `sentence`, and `label`.
- **test.csv**: Contains the test data with columns `ID`, `language`, and `sentence`.

## Model

The notebook uses the LLaMA model, specifically the `8b-instruct` version, which is fine-tuned for sentiment analysis tasks. The model is loaded with 4-bit quantization to reduce memory usage and improve efficiency.

### Fine-Tuning

The model is fine-tuned using the LoRA technique, which allows for efficient adaptation of large models with minimal additional parameters. The training process is configured with the following parameters:

- **LoRA Configuration**:
  - `lora_alpha`: 16
  - `lora_dropout`: 0.1
  - `r`: 16
  - `target_modules`: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

- **Training Arguments**:
  - `num_train_epochs`: 10
  - `per_device_train_batch_size`: 1
  - `gradient_accumulation_steps`: 4
  - `learning_rate`: 2e-4
  - `weight_decay`: 0.001
  - `fp16`: True
  - `max_grad_norm`: 0.3

## Usage

1. **Data Preparation**: The dataset is loaded and split into training and validation sets. The data is then formatted into prompts suitable for the LLaMA model.

2. **Model Training**: The LLaMA model is fine-tuned using the prepared dataset. The training process is logged using Weights & Biases (WandB) for monitoring.

3. **Evaluation**: The model's performance is evaluated on the validation set using F1 score, classification report, and confusion matrix.

4. **Prediction**: The fine-tuned model is used to predict the sentiment of sentences in the test set. The predictions are saved to a CSV file for submission.

## Results

The notebook provides detailed evaluation metrics, including the overall F1 score and class-specific F1 scores for positive and negative sentiments. The results are visualized using a confusion matrix and classification report.

## Saving Predictions

The final predictions for the test set are saved in a CSV file named `submission.csv`, which can be used for further analysis or submission to a competition.

## Conclusion

This notebook provides a comprehensive pipeline for fine-tuning the LLaMA model for multi-lingual sentiment analysis. By leveraging LoRA and PEFT, the model can be fine-tuned efficiently, even on limited hardware resources. The notebook also includes detailed evaluation and visualization tools to assess the model's performance.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Hugging Face for the Transformers library.
- Meta AI for the LLaMA model.
- Weights & Biases for experiment tracking.

---

For any questions or issues, please open an issue on the repository or contact the author.
