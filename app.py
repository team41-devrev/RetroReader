#import streamlit as st

import io
import os
import yaml
import pyarrow
import tokenizers
import pandas as pd
import datasets
from datasets import Dataset, DatasetDict
import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = "true"

# SETTING PAGE CONFIG TO WIDE MODE
#st.set_page_config(layout="wide")

#@st.cache
def from_library():
    from retro_reader import RetroReader
    from retro_reader import constants as C
    return C, RetroReader

C, RetroReader = from_library()



#@st.cache(hash_funcs=my_hash_func, allow_output_mutation=True)
def load_ko_electra_small_model():
    config_file = "configs/inference_ko_electra_small.yaml"
    return RetroReader.load(config_file=config_file)


# @st.cache(hash_funcs=my_hash_func, allow_output_mutation=True)
def load_en_electra_large_model():
     config_file = "configs/inference_en_electra_large.yaml"
     return RetroReader.load(config_file=config_file)


RETRO_READER_HOST = {
    # "klue/roberta-large": load_ko_roberta_large_model(),
    #"monologg/koelectra-small-v3-discriminator": load_ko_electra_small_model(),
    "google/electra-large-discriminator": load_en_electra_large_model(),
}


def main():
    #st.title("Retrospective Reader Demo")
    
    #st.markdown("## Model name")
    '''option = st.selectbox(
        label="Choose the model used in retro reader",
        options=(
            "[ko_KR] klue/roberta-large",
            "[ko_KR] monologg/koelectra-small-v3-discriminator",
            "[en_XX] google/electra-large-discriminator"
        ),
        index=1,
    )'''
    option = "[en_XX] google/electra-large-discriminator"
    lang_code, model_name = option.split(" ")
    
    retro_reader = RETRO_READER_HOST[model_name]
    
    # retro_reader = load_model()
    lang_prefix = "KO" if lang_code == "[ko_KR]" else "EN"
    height = 300 if lang_code == "[ko_KR]" else 200
    retro_reader.null_score_diff_threshold = 0.0
    retro_reader.rear_threshold = 0.0
    retro_reader.n_best_size = 20
    retro_reader.beta1 = 1.0
    retro_reader.beta2 = 1.0
    retro_reader.best_cof = 1.0
    return_submodule_outputs = False
    

    df_train = pd.read_csv("./test_stratified.csv")
    theme_train = df_train['Theme']
    paragraph_train = df_train['Paragraph']
    question_train = df_train['Question']
    answerPossible_train = df_train['Answer_possible']
    answerText_train = df_train['Answer_text']
    answerStart_train = df_train['Answer_start']

    id = []
    title = []
    context = []
    question = []
    answers = []

    for i in theme_train.keys():
        id.append(str(i))
        title.append(theme_train[i])
        context.append(paragraph_train[i])
        question.append(question_train[i])
        answerDict = {}
        if answerPossible_train[i]==False:
            answerDict['text'] = []
            answerDict['answer_start'] = []
        else:
            answerDict['text'] = [answerText_train[i][2:-2]]
            answerDict['answer_start'] = [int(answerStart_train[i][1:-1])]
        answers.append(answerDict)

    df1 = pd.DataFrame({'id': id, 'title': title, 'context' : context , 'question' : question , 'answers' : answers})

    DDD = DatasetDict()

    DDD['test'] = Dataset.from_pandas(df1)
    outputs = retro_reader.inference(DDD['test'])
    #answer, score = outputs[0]["id-01"], outputs[1]
    print("***************")
    print(outputs)
    print("**************")

    np.save("o1_zero_shot.npy",outputs)

    
if __name__ == "__main__":
    main()
