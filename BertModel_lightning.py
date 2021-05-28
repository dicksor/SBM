import torch
import pytorch_lightning as pl

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        text = encodings[idx]
        input = tokenize_tweet(text)
        item = {key: torch.tensor(val) for key, val in self.input.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def tokenize_tweet(tweet_text):
        return tokenizer(tweet_text, truncation=True, padding=True)
    
class BertDataModule (pl.LightningDataModule):
    def _init__(self,x_tr,y_tr,x_val,y_val,x_test,y_test,tokenizer, batch_size=16):
        super().__init__()
        self.tr_text = x_tr
        self.tr_label = y_tr
        self.val_text = x_val
        self.val_label = y_val
        self.test_text = x_test
        self.test_label = y_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self):
        self.train_dataset = Dataset(quest=self.tr_text,  tags=self.tr_label,tokenizer=self.tokenizer)
        self.val_dataset= Dataset(quest=self.val_text, tags=self.val_label,tokenizer=self.tokenizer)
        self.test_dataset =Dataset(quest=self.test_text, tags=self.test_label,tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size= self.batch_size, shuffle = True , num_workers=4)

    def val_dataloader(self):
        return DataLoader (self.val_dataset,batch_size= 16)

    def test_dataloader(self):
        return DataLoader (self.test_dataset,batch_size= 16)

class BertModel(pl.LightningModule):
    def __init__(self,n_classes=2,steps_per_epoch=None,n_epochs=3, lr=2e-5):
        super().__init__()
        config = AutoConfig.from_pretrained('bert-base-cased', num_labels=1)
        self.model = AutoModelForSequenceClassification.from_config(config)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, input_ids, attn_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        #output = self.classifier(output.pooler_output)
            
        return output

class SBMBertClassifier:
    def __init__(self):
        self.model = BertModel()
        self.trainer = pl.Trainer(max_epochs = 10)
    def train(datamodule):
        self.trainer = self.trainer.fit(self.model, datamodule)

    def validate(self):
        self.trainer.test()

    def test(self):
        self.trainer.test()

    def predict(self, text):
        # TODO ??? No predict in pl.trainer
        return self.model(text)

