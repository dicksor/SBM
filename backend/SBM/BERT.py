import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments
from sklearn.utils import class_weight
from datasets import load_metric
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification

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
        return self.tokenizer(tweet_text, truncation=True, padding=True, max_length=140)
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, tokenizer):
        self.tokenizer = tokenizer
        self.encodings = [self.tokenize_tweet(tweet) for tweet in encodings]

    def __getitem__(self, idx):
        text = self.encodings[idx]
        item = {key: torch.tensor(val) for key, val in text.items()}
        return item

    def __len__(self):
        return len(self.encodings)

    def tokenize_tweet(self, tweet_text):
        return self.tokenizer(tweet_text, truncation=True, padding=True, max_length=140)
    
class BertDataModule():
    def __init__(self,x_tr,y_tr,x_test,y_test,tokenizer, batch_size=16):
        super().__init__()
        self.tr_text = x_tr
        self.tr_label = y_tr
        self.test_text = x_test
        self.test_label = y_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.class_weight = None

        self.setup()

    def setup(self):
        self.train_dataset = Dataset(encodings=self.tr_text,  labels=self.tr_label,tokenizer=self.tokenizer)
        #self.val_dataset= Dataset(encodings=self.val_text, labels=self.val_label,tokenizer=self.tokenizer)
        self.test_dataset =Dataset(encodings=self.test_text, labels=self.test_label,tokenizer=self.tokenizer)
        self.compute_class_weight(self.tr_label.tolist() + self.test_label.tolist())

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size= self.batch_size, shuffle = True , num_workers=4)

    def val_dataloader(self):
        return DataLoader (self.val_dataset,batch_size= 16)

    def test_dataloader(self):
        return DataLoader (self.test_dataset,batch_size= 16)

    def compute_class_weight(self, y):
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        self.class_weight = torch.tensor(cw, dtype=torch.float)

#from transformers import Trainer
class SBMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss

class SBMBertClassifier:
    def __init__(self, n_epochs=3, lr=2e-5, batch_size=16, tokenizer=None):
        config = AutoConfig.from_pretrained('bert-base-cased', num_labels=1)
        self.model = AutoModelForSequenceClassification.from_config(config)
        self.bertscore = load_metric("bertscore")
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.trainer = None
        self.test_trainer = None
        self.tokenizer = tokenizer

    # def compute_metrics(self, p):
    #     pred, labels = p
    #     pred = np.argmax(pred, axis=1)

    #     accuracy = accuracy_score(y_true=labels, y_pred=pred)
    #     recall = recall_score(y_true=labels, y_pred=pred)
    #     precision = precision_score(y_true=labels, y_pred=pred)
    #     f1 = f1_score(y_true=labels, y_pred=pred)

    #     return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            # 'test': f1_score(y_true=labels, y_pred=pred)
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        print(loss)
        return (loss, outputs) if return_outputs else loss

    def train(self, datamodule):
        args = TrainingArguments(
            output_dir="sbm",
            evaluation_strategy = "epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.n_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
        )
        self.trainer = Trainer(
            self.model,
            args,
            train_dataset=datamodule.train_dataset,
            eval_dataset=datamodule.test_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics
        )
        # self.trainer.compute_loss = self.compute_loss
        self.trainer.train()
        # Compute the final loss with class weight
        
        # input_ids = [torch.tensor(x['input_ids']) for x in datamodule.train_dataset.encodings]
        # input_ids = pad_sequence(input_ids)
        # #att_mask = torch.tensor([x['attention_mask'] for x in datamodule.train_dataset.encodings])
        # labels = [torch.tensor(x) for x in datamodule.train_dataset.labels]
        # output = self.model(input_ids=input_ids, labels=labels)
        # print(output.logits)
        # loss = loss_fn(np.array(output.logits), np.array(labels))
        # loss.backward()

    def score():
        metric = load_metric("bertscore")
        for batch in dataset:
            inputs, references = batch
            predictions = model(inputs)
            metric.add_batch(predictions=predictions, references=references)
        score = metric.compute()
        print(score)

    def test(self, test_dataset, filename=None):
        # Load trained model
        model_path = None
        if filename : 
            model_path = filename
        else : 
          model_path = "sbm/checkpoint-" + str(self.n_epochs)
          
        config = AutoConfig.from_pretrained(model_path, num_labels=1)
        model = AutoModelForSequenceClassification.from_config(config)
        # model = BertForSequenceClassification.from_pretrained(model_path, num_labels=1) 
        # Define test trainer
        self.test_trainer = Trainer(model, tokenizer=test_dataset.tokenizer, compute_metrics=self.compute_metrics)
        # Make prediction
        # for e in test_dataset.encodings:
        #     print(e.input_ids)
        #     e.input_ids = pad_sequence(torch.tensor(e.input_ids), False, 1)
        raw_pred, _, metrics = self.test_trainer.predict(test_dataset) 
        # Preprocess raw predictions
        # metrics = self.compute_metrics(raw_pred)
        y_pred = np.argmax(raw_pred, axis=1)
        return y_pred, metrics

    def predict(self, texts):
        pred, metrics = self.test(TestDataset(texts, self.tokenizer))
        return pred

