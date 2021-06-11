import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments
from datasets import load_metric
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, tokenizer):
        self.tokenizer = tokenizer
        self.encodings = [self.tokenize_tweet(tweet) for tweet in encodings]
        self.labels = labels

    def __getitem__(self, idx):
        text = self.encodings[idx]
        item = {key: torch.tensor(val) for key, val in text.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def tokenize_tweet(self, tweet_text):
        return self.tokenizer(tweet_text, truncation=True, padding=True)

class BertDataModule():
    def __init__(self,x_tr,y_tr,x_test,y_test,tokenizer, batch_size=16):
        super().__init__()
        self.tr_text = x_tr
        self.tr_label = y_tr
        self.test_text = x_test
        self.test_label = y_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.setup()

    def setup(self):
        self.train_dataset = Dataset(encodings=self.tr_text,  labels=self.tr_label,tokenizer=self.tokenizer)
        #self.val_dataset= Dataset(encodings=self.val_text, labels=self.val_label,tokenizer=self.tokenizer)
        self.test_dataset =Dataset(encodings=self.test_text, labels=self.test_label,tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size= self.batch_size, shuffle = True , num_workers=4)

    def val_dataloader(self):
        return DataLoader (self.val_dataset,batch_size= 16)

    def test_dataloader(self):
        return DataLoader (self.test_dataset,batch_size= 16)

class SBMBertClassifier:
    def __init__(self, n_epochs=3, lr=2e-5, batch_size=16):
        config = AutoConfig.from_pretrained('bert-base-cased', num_labels=1)
        self.model = AutoModelForSequenceClassification.from_config(config)
        self.bertscore = load_metric("bertscore")
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def compute_metrics(self, eval_pred):
        acc_metric = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        #self.bertscore.add_batch(predictions=predictions, references=labels)
        return acc_metric.compute(predictions=predictions, references=labels)

    def train(self, datamodule):
        args = TrainingArguments(
            "sbm",
            evaluation_strategy = "epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.n_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            self.model,
            args,
            train_dataset=datamodule.train_dataset,
            eval_dataset=datamodule.test_dataset,
            tokenizer=datamodule.tokenizer,
            compute_metrics=self.compute_metrics)
        trainer.train()

    def score():
        metric = load_metric("bertscore")
        for batch in dataset:
            inputs, references = batch
            predictions = model(inputs)
            metric.add_batch(predictions=predictions, references=references)
        score = metric.compute()
        print(score)

    def validate(self):
        self.trainer.test()

    def test(self):
        self.trainer.test()

    def predict(self, text):
        # TODO ??? No predict in pl.trainer
        return self.model(text)