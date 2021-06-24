import pandas as pd 
import re
import gensim 
from gensim.parsing.preprocessing import remove_stopwords
import numpy
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity 
import gensim.downloader as api

def retrieveAndPrintFAQAnswer(question_embedding,sentence_embeddings,FAQdf,sentences):
    max_sim=-1 
    index_sim=-1 
    for index,faq_embedding in enumerate(sentence_embeddings):
        sim=cosine_similarity(faq_embedding,question_embedding)[0][0] 
        print(index, sim, sentences[index])
        if sim>max_sim:
            max_sim=sim 
            index_sim=index 
       
    print("\n")
    print("Question: ",question)
    print("\n") 
    print("Retrieved: ",FAQdf.iloc[index_sim,0]) 
    print(FAQdf.iloc[index_sim,1])   

def clean_sentence(sentence, stopwords=False):
    
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
    
    if stopwords:
         sentence = remove_stopwords(sentence)
    
    return sentence
                    
def get_cleaned_sentences(df,stopwords=False):    
    sents=df[["Questions"]] 
    cleaned_sentences=[]

    for index,row in df.iterrows():
        cleaned=clean_sentence(row["Questions"],stopwords) 
        cleaned_sentences.append(cleaned) 
    return cleaned_sentences 

def getWordVec(word,model):
        samp=model['computer'] 
        vec=[0]*len(samp) 
        try:
                vec=model[word] 
        except:
                vec=[0]*len(samp) 
        return (vec)


def getPhraseEmbedding(phrase,embeddingmodel):
                       
        samp=getWordVec('computer', embeddingmodel) 
        vec=numpy.array([0]*len(samp)) 
        den=0 
        for word in phrase.split():
            den=den+1 
            vec=vec+numpy.array(getWordVec(word,embeddingmodel))  
        return vec.reshape(1, -1)

#precrocessing
df=pd.read_csv("KafkaFAQ.csv") 
df.columns=["Questions","Answers"] 

cleaned_sentences=get_cleaned_sentences(df,stopwords=True)
cleaned_sentences_with_stopwords=get_cleaned_sentences(df,stopwords=False)

#input question
question_orig="how is this priced"
question=clean_sentence(question_orig,stopwords=False) 

#load glove model

model=None 
try:
    model = gensim.models.KeyedVectors.load("./w2vecmodel.mod")
    print("Loaded model")
except:            
    model = api.load('word2vec-google-news-300')
    model.save("./w2vecmodel.mod")
    print("Saved model")    
glove_embedding_size=len( model['computer']) 

# predict the faq

sent_embeddings=[]
for sent in cleaned_sentences:
    sent_embeddings.append(getPhraseEmbedding(sent, model))
    
question_embedding=getPhraseEmbedding(question, model)
retrieveAndPrintFAQAnswer(question_embedding,sent_embeddings,df, cleaned_sentences_with_stopwords)
