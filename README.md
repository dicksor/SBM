# SBM
## Project files
The project in its entirety can be downloaded on the following link : https://www.swisstransfer.com/d/afb063a9-338e-44e7-afd6-545821e4a055

## Report
The report is is folder admin/

## Application setup
First install requierements with command : ``pip install -r requierements.txt``. 

Go on the backend folder and then run : ``set FLASK_APP=app``

Start the backend with : ``flask run``

Go on the folder frontend and open file : ``index.html``

## Files
* SBM : TextProcessor.py : Textproccesor class
* admin : report and schemas
* backend : Backend files
* frontend : Frontend files
* BertModel_lightning.py : BERT with abstraction librairies
* Bert.ipynb : Sentiment analysis with BERT
* ClassicML.ipynb : Sentiment analysis with classic ML methods, like SVM, Random Forest, ...
* RNN_with_GloVe.ipynb : Sentiment analysis with RNN and GloVe
* RNN_without_GloVe.ipynb : Sentiment analysis with RNN only 