import spacy

from sklearn.base import BaseEstimator, TransformerMixin
import preprocessor as tp

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

SpecialTokenMethod = enum('KEEP', 'REMOVE', 'PREPROCESS')


class TextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stop_word=False, remove_punctuation=False, min_word_size=2, twitter_tokens=None, special_token_method=SpecialTokenMethod.PREPROCESS):
        
        self.remove_stop_word = remove_stop_word
        self.min_word_size = min_word_size
        self.remove_punctuation = remove_punctuation
        self.vocab = set()
        self.nlp = spacy.load('en_core_web_sm')
        self.twitter_tokens = twitter_tokens
        
        if self.twitter_tokens is None :
            tp.set_options(*[tp.OPT.URL, tp.OPT.MENTION, tp.OPT.HASHTAG, tp.OPT.RESERVED, tp.OPT.EMOJI, tp.OPT.SMILEY, tp.OPT.NUMBER, tp.OPT.ESCAPE_CHAR])
        else :
            tp.set_options(*self.twitter_tokens)
            
        self.special_token_method = special_token_method
        
    def __spacy_text_processing(self, sentence):
        final_sentence = []
        for word in self.nlp(sentence):
                        
            if self.remove_stop_word and word.is_stop:
                continue
                    
            if self.min_word_size!=-1 and len(word.text) < self.min_word_size:
                continue
                    
            if self.remove_punctuation and word.is_punct:
                continue
                    
            if word.text[-1] != '$':
                final_sentence.append(word.lemma_.lower())
                self.vocab.add(word.lemma_.lower())
            else : 
                final_sentence.append("$" + word.text.upper())
            
        return final_sentence
    
    def __twitter_preprocess(self, sentence):
        if self.special_token_method == SpecialTokenMethod.KEEP:
            return sentence
        if self.special_token_method == SpecialTokenMethod.REMOVE:
            return tp.clean(sentence)
        if self.special_token_method == SpecialTokenMethod.PREPROCESS:
            return tp.tokenize(sentence)
        else:
            raise ValueError('This special token method doesn\'t exist !')
        
    def transform(self, X, y=None):
        X_transformed = []
        for sentence in X:
            twitter_sentence = self.__twitter_preprocess(sentence)
            X_transformed.append(' '.join(self.__spacy_text_processing(twitter_sentence)))
        return X_transformed
    
    def fit(self, X, y=None):
        return self