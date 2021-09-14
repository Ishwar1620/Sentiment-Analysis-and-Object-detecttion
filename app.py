import tensorflow as tf
from PIL import Image
from flask import Flask, render_template
from flask import request
import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path 
from typing import List
from multiprocessing import Process, freeze_support
import pickle
import os
import urllib.request
import torch
import torch.optim as optim
from flask_restful import reqparse, abort, Api, Resource
from flask import jsonify
from flask import make_response
import random 
import spacy
from Model import final2
app = Flask(__name__)
api = Api(app)

# fastai
from fastai import *
from fastai.text import *
from fastai.callbacks import *

# transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig


train = pd.read_csv('sample.tsv',sep="\t")
test = pd.read_csv('sample.tsv',sep="\t")

MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)
}
# Parameters
seed = 42
use_fp16 = False
bs = 16

model_type = 'roberta'
pretrained_model_name = 'roberta-base'

model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]

def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

seed_all(seed)

class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens

transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])

class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:
        "Convert a list of `nums` to their tokens."
        nums = np.array(nums).tolist()
        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
    
    def __getstate__(self):
        return {'itos':self.itos, 'tokenizer':self.tokenizer}

    def __setstate__(self, state:dict):
        self.itos = state['itos']
        self.tokenizer = state['tokenizer']
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)

transformer_processor = [tokenize_processor, numericalize_processor]

pad_first = bool(model_type in ['xlnet'])
pad_idx = transformer_tokenizer.pad_token_id

databunch = (TextList.from_df(train, cols='Phrase', processor=transformer_processor)
             .split_by_rand_pct(0.1,seed=seed)
             .label_from_df(cols= 'Sentiment')
             .add_test(test)
             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))

# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=pad_idx).type(input_ids.type()) 
        
        logits = self.transformer(input_ids,
                                  attention_mask = attention_mask)[0]   
        return logits
    
config = config_class.from_pretrained(pretrained_model_name)
config.num_labels = 3
config.use_bfloat16 = use_fp16

transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)

custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)
detector_output,image_string_placeholder,decoded_image,init_ops,sess,draw_boxes=final2()

from fastai.callbacks import *
from transformers import AdamW
from functools import partial

CustomAdamW = partial(AdamW, correct_bias=False)

learner_sentiment = Learner(databunch, 
                  custom_transformer_model, 
                  opt_func = CustomAdamW, 
                  metrics=[accuracy, error_rate])

learner_emotional = Learner(databunch, 
                  custom_transformer_model, 
                  opt_func = CustomAdamW, 
                  metrics=[accuracy, error_rate])

# Show graph of learner stats and metrics after each epoch.
learner_emotional.callbacks.append(ShowGraph(learner_sentiment))
learner_sentiment.callbacks.append(ShowGraph(learner_emotional))

# Put learn in FP16 precision mode. --> Seems to not working
if use_fp16: learner = learner_sentiment.to_fp16()
if use_fp16: learner = learner_emotional.to_fp16()
    
list_layers = [learner_sentiment.model.transformer.roberta.embeddings,
              learner_sentiment.model.transformer.roberta.encoder.layer[0],
              learner_sentiment.model.transformer.roberta.encoder.layer[1],
              learner_sentiment.model.transformer.roberta.encoder.layer[2],
              learner_sentiment.model.transformer.roberta.encoder.layer[3],
              learner_sentiment.model.transformer.roberta.encoder.layer[4],
              learner_sentiment.model.transformer.roberta.encoder.layer[5],
              learner_sentiment.model.transformer.roberta.encoder.layer[6],
              learner_sentiment.model.transformer.roberta.encoder.layer[7],
              learner_sentiment.model.transformer.roberta.encoder.layer[8],
              learner_sentiment.model.transformer.roberta.encoder.layer[9],
              learner_sentiment.model.transformer.roberta.encoder.layer[10],
              learner_sentiment.model.transformer.roberta.encoder.layer[11],
              learner_sentiment.model.transformer.roberta.pooler]

list_layers = [learner_emotional.model.transformer.roberta.embeddings,
              learner_emotional.model.transformer.roberta.encoder.layer[0],
              learner_emotional.model.transformer.roberta.encoder.layer[1],
              learner_emotional.model.transformer.roberta.encoder.layer[2],
              learner_emotional.model.transformer.roberta.encoder.layer[3],
              learner_emotional.model.transformer.roberta.encoder.layer[4],
              learner_emotional.model.transformer.roberta.encoder.layer[5],
              learner_emotional.model.transformer.roberta.encoder.layer[6],
              learner_emotional.model.transformer.roberta.encoder.layer[7],
              learner_emotional.model.transformer.roberta.encoder.layer[8],
              learner_emotional.model.transformer.roberta.encoder.layer[9],
              learner_emotional.model.transformer.roberta.encoder.layer[10],
              learner_emotional.model.transformer.roberta.encoder.layer[11],
              learner_emotional.model.transformer.roberta.pooler]

learner_emotional.split(list_layers)
learner_sentiment.split(list_layers)

path = ''
learner_sentiment = load_learner(path, file = 'model_sentiment.pkl')
learner_emotional = load_learner(path, file = 'model_emotional.pkl')



class Home(Resource):
    def get(self):
        return make_response(render_template('index.html'))

class Sentiment_Input(Resource):
    def get(self):
        return make_response(render_template('sentiment_input.html'))


class Emotion_Input(Resource):
    def get(self):
        return make_response(render_template('emotion_input.html'))

class NLP_Input(Resource):
    def get(self):
        return make_response(render_template('nlp_input.html'))

class Object_Input(Resource):
    def get(self):
        return make_response(render_template('Object_detection.html'))

class Object_Output(Resource):
    def get(self):
        return make_response(render_template('image.html'))

class Object_Detection(Resource):

    def post(self):
        def model_detect(img_path):
            sample_image_path = img_path  

                # Load our sample image into a binary string
            with tf.compat.v1.gfile.Open(sample_image_path, "rb") as binfile:
                    image_string = binfile.read()

                # Run the graph we just created
            sess.run(init_ops)
            result_out, image_out = sess.run(
                        [detector_output, decoded_image],feed_dict={image_string_placeholder: image_string}
                    )
            
            image_with_boxes = draw_boxes(
            np.array(image_out), result_out["detection_boxes"],
            result_out["detection_class_entities"], result_out["detection_scores"])
        
            return image_with_boxes
        if request.method == 'POST':
            f = request.files['file']
            f.save(f.filename)
            print("save sucessfully")
            print(type(f.filename))
            preds = model_detect(f.filename)
            data = Image.fromarray(preds)
            data.save("static/preds.jpg")
            filename=f.filename
            print(filename)
            return make_response(render_template("image.html"))
        

class Sentiment_Analysis(Resource):

    def post(self):
        if request.method == 'POST':
            # print(request)
            phrase = request.form.get('phrase')
            print(phrase)
            prediction = learner_sentiment.predict(str(phrase))
            result = int(prediction[0])

            print(result)      
            if int(result)== 0:
                output ='Negative'
            elif int(result)==1:
                output ='Neutral'  
            elif int(result)==2:
                output ='Positive'  

            return_obj = {'text': str(phrase) ,'prediction':output,'status':200 , 'message':'Successfully predicted'}
            return make_response(render_template("result.html",prediction=return_obj))

class Emotion_Analysis(Resource):

    def post(self):
        if request.method == 'POST':
            # print(request)
            phrase = request.form.get('phrase')
            print(phrase)
            prediction = learner_emotional.predict(str(phrase))
            result = int(prediction[0])
            print(result)      
            if int(result)== 0:
                output ='Joy'
            elif int(result)==1:
                output =' Sadness'  
            elif int(result)==2:
                output ='Fear'  
            elif int(result)==3:
                output = 'Anger' 
            elif int(result)==4:
                output = 'Love'  
            elif int(result)==5:
                output = 'Surprise'  

            return_obj = {'text': str(phrase) ,'prediction':output,'status': 200, 'message':'Successfully predicted'}
            return make_response(render_template("result.html",prediction=return_obj))

class POS(Resource):
    def post(self):
        nlp = spacy.load("en_core_web_sm")
        data = request.form.get('phrase')
        print(data)
        doc = nlp(data)
        output = {}
        for token in doc:
            if token.pos_ not in output:
                output[token.pos_] = list()
            output[token.pos_].append(str(token.text))


        # output = jsonify(output)
        print(output)     
        return make_response(render_template("result.html",prediction=output))

class NER(Resource):
    def post(self):
        nlp = spacy.load("en_core_web_sm")
        data = request.form.get('phrase')
        print(data)
        print(request)
        doc = nlp(data)
        output = {}
        for ent in doc.ents:

            if ent.label_ not in output:
                output[ent.label_] = list()
            output[ent.label_].append(str(ent.text))
        print(output)
        # output = jsonify(output)
        
        return make_response(render_template("result.html",prediction=output))


class Get_Sentiment(Resource):
    def post(self):

        phrase = request.get_json()
        prediction = learner_sentiment.predict(str(phrase))
        result = int(prediction[0])

        print(result)      
        if int(result)== 0:
            output ='Negative'
        elif int(result)==1:
            output ='Neutral'  
        elif int(result)==2:
            output ='Positive'  

        return output

class Get_Emotion(Resource):
    def post(self):

        phrase = request.get_json()
        prediction = learner_emotional.predict(str(phrase))
        result = int(prediction[0])

        print(result)      
        print(result)      
        if int(result)== 0:
            output ='Joy'
        elif int(result)==1:
            output =' Sadness'  
        elif int(result)==2:
            output ='Fear'  
        elif int(result)==3:
            output = 'Anger' 
        elif int(result)==4:
            output = 'Love'  
        elif int(result)==5:
            output = 'Surprise'  

        return output
api.add_resource(Object_Detection,'/object-detection')
api.add_resource(Sentiment_Analysis, '/sentiment-analysis')
api.add_resource(Emotion_Analysis, '/emotion-analysis')
api.add_resource(POS, '/POS')
api.add_resource(NER, '/NER')
api.add_resource(Home, '/')
api.add_resource(Object_Input,'/object_input')
api.add_resource(Sentiment_Input, '/sentiment_input')
api.add_resource(Emotion_Input, '/emotion_input')
api.add_resource(NLP_Input, '/nlp_input')

api.add_resource(Get_Sentiment, '/get-sentiment')
api.add_resource(Get_Emotion, '/get-emotion')
api.add_resource(Object_Output,'/object-output')
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5001,debug=True)
