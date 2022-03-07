import torch
from transformers import T5Tokenizer, AutoTokenizer, T5ForConditionalGeneration
import pandas as pd
from fuzzywuzzy import process

from .util import generate_answer_text, generate_unifiedqa_text

class QuestionCollection(object):
    '''
    Collection of questions
    :params
        question_file: question file path, csv format
    '''
    def __init__(self, question_file:str):
        #load question
        self.question_file = question_file
        self.question_list = [] # question in text
        self.answer_list = [] # answer in text

        self.raw_answer_list = [] # answer in choice
        self.load_questions()

    def load_questions(self):
        '''
        load questions
        :return:
            a list containing questions and answers
        '''
        df = pd.read_csv(self.question_file)
        for i in range(len(df)):
            question_type = df.iloc[i][0]
            question = df.iloc[i][1]
            if question_type == "Multiple-choice":
                raw_answer = df.iloc[i][2].split(",")
            else: #question_type == "Yes-no":
                raw_answer = ["yes","no"]
            
            self.raw_answer_list.append(raw_answer)

            answer = generate_answer_text(raw_answer, add_change_line=False)
            self.question_list.append(question.lower())
            self.answer_list.append(answer.lower())

    def __len__(self):
        return len(self.question_list)

class QAMachine(object):
    '''
    A machine to hold dataset and qa-model to perform question and answering
    '''
    def __init__(self, question_collection_file:str, token_model_name = "allenai/unifiedqa-t5-large",
                model_name:str="allenai/unifiedqa-t5-large"):
        '''
        :params:
            question_collection_file: the name of the question file
            dataset_name: the name of the dataset listed in #from datasets import list_datasets
            model_name: the name of the model in UnifiedQA
        '''
        # device
        self.use_cuda = True and torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(0) if self.use_cuda else "cpu")

         #load question collection
        self.question_collection = QuestionCollection(question_collection_file)

        # load model
        self.token_model_name = token_model_name
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        print("load model......")
        self.tokenizer = T5Tokenizer.from_pretrained(self.token_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        if self.use_cuda:
            self.model = self.model.to(self.device)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer(input_string, padding=True, truncation=True, max_length=100, return_tensors="pt").input_ids
        if self.use_cuda:
            input_ids = input_ids.to(self.device)   
        res = self.model.generate(input_ids, **generator_args)
        return [self.tokenizer.decode(x) for x in res]

    def predict_multi(self, text:str, question_list:list, answer_list:list, raw_answer_list:list):
        '''
        predict multiple qa
        '''
        assert len(question_list) == len(answer_list) == len(raw_answer_list)
        
        #print("Datasets QAMachine conduct survey on question {} : {}".format(str(question_id), 
        #    self.question_collection.question_answer_list[question_id][0]))
        batch_sentences = []
        for question, answer in zip(question_list, answer_list):
            qa_text = generate_unifiedqa_text(question, answer, text)
            batch_sentences.append(qa_text)


        question_answers = self.run_model(batch_sentences)
        #print(question_answers)

        answer_choices = []
        for question, qa, raw_answer in zip(question_list, question_answers, raw_answer_list):
            answer_choice = process.extractOne(qa, raw_answer)[0]
            # answer_index = self.question_collection.raw_answer_list[question_id].index(answer_choice)
            # print(question, text, answer_choice)
            answer_choices.append(answer_choice)

        return answer_choices
    