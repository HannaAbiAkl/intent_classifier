from pathlib import Path
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    default_data_collator,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, Value
import random
import logging
import sys
import os
import torch

def encode_intents(json_dataset):
     df = pd.json_normalize(json_dataset, record_path =['sentences'])
     # keep only relevant columns
     df = df[['text','intent']]
     # undersample data to equal least frequent class label
     # get count of fewest intent
     min_count_label = df.intent.value_counts().min()
     # get all intents
     label_types = list(df.intent.unique())
     subdatasets = list()
     for label_type in label_types:
          dataset_label_type = df[df['intent']==label_type]
          dataset_label_type = dataset_label_type.sample(min_count_label)
          subdatasets.append(dataset_label_type)
     dataset_undersampled = pd.concat(subdatasets)
     # randomize final dataset
     dataset_undersampled = dataset_undersampled.sample(frac=1)
     le = LabelEncoder()
     # tranform json data to dataframe and encode intent values
     df_data = pd.DataFrame({'sentence': dataset_undersampled['text'],
                              'intent': dataset_undersampled['intent'],
                              'label': le.fit_transform(dataset_undersampled['intent'])})
     return df_data

def convert_labels(dataset):
     # cast label values from int to float to avoid Trainer problems
     new_features = dataset.features.copy()
     new_features["label"] = Value('float64')
     dataset = dataset.cast(new_features)
     return dataset 

def trainIntentClassifier(bot_id, model_name, dataset):
     # set up logging
     logger = logging.getLogger(__name__)

     logging.basicConfig(
          level=logging.getLevelName("INFO"),
          handlers=[logging.StreamHandler(sys.stdout)],
          format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
     )

     # create model path for bot id
     model_dir = "resources/" + "models/" + model_name + "/" + bot_id
     Path(model_dir).mkdir(parents=True, exist_ok=True)

     # load datasets
     df_dataset = encode_intents(dataset)
     train_dataset, test_dataset = train_test_split(df_dataset, test_size=0.15, random_state=0, stratify=df_dataset.label.values)
     raw_train_dataset = convert_labels(Dataset.from_pandas(train_dataset))
     raw_test_dataset = convert_labels(Dataset.from_pandas(test_dataset))

     # load tokenizer
     tokenizer = AutoTokenizer.from_pretrained(model_name)

     # preprocess function, tokenizes text
     def preprocess_function(examples):
          return tokenizer(examples["sentence"], padding="max_length", truncation=True)

     # preprocess dataset
     train_dataset = raw_train_dataset.map(
          preprocess_function,
          batched=True,
     )
     test_dataset = raw_test_dataset.map(
          preprocess_function,
          batched=True,
     )

     # define labels
     num_labels = len(train_dataset.unique("label"))

     # print size
     logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
     logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

     # compute metrics function for binary classification
     def compute_metrics(pred):
          labels = pred.label_ids
          preds = pred.predictions.argmax(-1)
          precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
          acc = accuracy_score(labels, preds)
          return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

     # download model from model hub
     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

     # define training args
     training_args = TrainingArguments(
          output_dir=model_dir,
          num_train_epochs=1,
          per_device_train_batch_size=32,
          per_device_eval_batch_size=64,
          warmup_steps=500,
          fp16=True,
          evaluation_strategy="epoch",
          save_strategy="epoch",
          logging_dir=f"{model_dir}/logs",
          learning_rate=float(3e-5),
          load_best_model_at_end=True,
          metric_for_best_model="f1",
     )

     # create Trainer instance
     trainer = Trainer(
          model=model,
          args=training_args,
          compute_metrics=compute_metrics,
          train_dataset=train_dataset,
          eval_dataset=test_dataset,
          tokenizer=tokenizer,
          data_collator=default_data_collator,
     )

     # train model
     trainer.train()

     # evaluate model
     eval_result = trainer.evaluate(eval_dataset=test_dataset)

     # writes eval result to file which can be accessed later in s3 ouput
     with open(os.path.join(model_dir, "eval_results.txt"), "w") as writer:
          print(f"***** Eval results *****")
          for key, value in sorted(eval_result.items()):
               writer.write(f"{key} = {value}\n")

     # update the config for prediction
     intent_labels = df_dataset.intent.unique()
     label2id = {}
     id2label = {}
     for i in range(len(intent_labels)):
          label2id[intent_labels[i]] = i
          id2label[i] = intent_labels[i]
     
     trainer.model.config.label2id = label2id
     trainer.model.config.id2label = id2label

     # Saves the model to directory
     model.save_pretrained(model_dir)
     tokenizer.save_pretrained(model_dir)


def predictIntentClassifier(model_dir, dataset):
     # load model
     model = AutoModelForSequenceClassification.from_pretrained(model_dir)
     tokenizer = AutoTokenizer.from_pretrained(model_dir)
     # get prediction text
     df = pd.json_normalize(dataset, record_path =['sentences'])
     # keep only relevant columns
     df = df[['text']]
     df.rename(columns={'text': 'sentence'}, inplace=True)
     raw_test_dataset = Dataset.from_pandas(df)
     # preprocess function, tokenizes text
     def preprocess_function(examples):
          return tokenizer(examples["sentence"], padding="max_length", truncation=True)

     # preprocess dataset
     test_dataset = raw_test_dataset.map(
          preprocess_function,
          batched=True,
     )
     # Define prediction Trainer
     trainer = Trainer(model)
     # predict intents
     raw_pred, _, _ = trainer.predict(test_dataset)
     # Preprocess raw predictions to get class id
     y_pred = np.argmax(raw_pred, axis=1)
     # map predicted class id to label
     predictions = [trainer.model.config.id2label[y_elem] for y_elem in y_pred]
     return predictions