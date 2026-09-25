import mlflow
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

mlflow.set_experiment("Intermediate_AI_Model_Finetuning")

with mlflow.start_run():
    model_name = "prajjwal1/bert-tiny"
    mlflow.log_param("base_model", model_name)
    mlflow.log_param("epochs", 1)
    
    data = {
        "text": ["Great API response!", "Terrible latency performance.", "Excellent accuracy.", "Poor error handling."],
        "label": [1, 0, 1, 0]
    }
    dataset = Dataset.from_dict(data)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize_func(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    
    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        use_cpu=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )
    
    train_result = trainer.train()
    mlflow.log_metric("train_loss", train_result.training_loss)
    
    print("\n--------------------------------------------------")
    print("SUCCESS: Model fine-tuning complete and tracked in MLflow!")
    print("--------------------------------------------------")
