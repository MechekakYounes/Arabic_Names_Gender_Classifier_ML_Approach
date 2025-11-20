from arabert.preprocess import ArabertPreprocessor
import pandas as pd 
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_excel("all_arabic_names_preprocessed.xlsx")
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})


model_name = "aubmindlab/bert-base-arabertv2"

arabert_prep = ArabertPreprocessor(
    model_name=model_name,
    apply_farasa_segmentation=False,  
)
df["Text"] = df["Name"].apply(lambda x: arabert_prep.preprocess(str(x)))
dataset = Dataset.from_pandas(df.rename(columns={"Text": "text", "Gender": "label"}))
dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=16)

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

args = TrainingArguments(
    output_dir="arabert_gender",
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

