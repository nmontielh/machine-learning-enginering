import pandas as pd
import string
import pprint

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer


################################################

def bagOfwords2():
    # Colocamos el bag of words, tomamos el texto y contamos la frecuencia de reccurrencia
    # comvertimos el documento en una matrix (matriz de frecuencia de distribucion)
    
    documents = ['Hello, how are you!',
                 'Win money, win from home.',
                 'Call me now.',
                 'Hello, Call hello you tomorrow?']
    
    # esta se puede utilizar 
    #bagOfWords(documents)
    # Convertimos el texto en numero como un bagOfWords
    # si agregamos stop_words = english debemos agregar LazyCorpusLoader
    count_vector = CountVectorizer()
    print(count_vector)
    
    count_vector.fit(documents)
    features_names = count_vector.get_feature_names()
    print(features_names)
    
    # creamos una matrix con los 4 documentos, documento con frecuencua de recurremcia
    doc_array = count_vector.transform(documents).toarray()
    print(doc_array)
    
    # convertimos en dataframes y colocamos el nombre de las columnas
    frecuency_matrix = pd.DataFrame(doc_array, columns=features_names)
    
    print(frecuency_matrix)
    

def bagOfWords(documents):
    # convertimos en minusculas todo el documento
    lower_case_documents = []
    for i in documents:
        lower_case_documents.append(i.lower())
    print(lower_case_documents)
    
    #removiendo la puntiacion
    sans_punctuation_documents = []
    
    for i in lower_case_documents:
        sans_punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
    print(sans_punctuation_documents)
    
    
    # Tokennizacion
    preprocessed_documents = []
    for i in sans_punctuation_documents:
        preprocessed_documents.append(i.split(' '))
    print(preprocessed_documents)
    
    
    #contamos con la frecuencia
    frequency_list = []
    
    for i in preprocessed_documents:
        frequency_counts = Counter(i)
        frequency_list.append(frequency_counts)
    pprint.pprint(frequency_list)


def bayes_teorem():
    '''
    Instructions:
    Calculate probability of getting a positive test result, P(Pos)
    '''
    # P(D)
    p_diabetes = 0.01
    
    # P(~D)
    p_no_diabetes = 0.99
    
    # Sensitivity or P(Pos|D)
    p_pos_diabetes = 0.9
    
    # Specificity or P(Neg/~D)
    p_neg_no_diabetes = 0.9
   

    '''
    Instructions:
    Compute the probability of an individual having diabetes, given that, that individual got a positive test result.
    In other words, compute P(D|Pos).
    
    The formula is: P(D|Pos) = (P(D) * P(Pos|D) / P(Pos)
    '''    
    # P(Pos)
    p_pos = (p_diabetes * p_pos_diabetes) + (p_no_diabetes * (1 - p_neg_no_diabetes))
    print('The probability of getting a positive test result P(Pos) is: {}',format(p_pos))
    
    # P(D|Pos)
    p_diabetes_pos = (p_diabetes * p_pos_diabetes) / p_pos
    print('Probability of an individual having diabetes, given that that individual got a positive test result is:\
    ',format(p_diabetes_pos))
    

    '''
    Instructions:
    Compute the probability of an individual not having diabetes, given that, that individual got a positive test result.
    In other words, compute P(~D|Pos).
    
    The formula is: P(~D|Pos) = (P(~D) * P(Pos|~D) / P(Pos)
    
    Note that P(Pos/~D) can be computed as 1 - P(Neg/~D). 
    
    Therefore:
    P(Pos/~D) = p_pos_no_diabetes = 1 - 0.9 = 0.1
    '''    
    # P(Pos/~D)
    p_pos_no_diabetes = 0.1
    
    # P(~D|Pos)
    p_no_diabetes_pos = (p_no_diabetes * p_pos_no_diabetes) / p_pos
    print ('Probability of an individual not having diabetes, given that that individual got a positive test result is:',p_no_diabetes_pos)


def naive_bayes():
    '''
    Instructions: Compute the probability of the words 'freedom' and 'immigration' being said in a speech, or
    P(F,I).
    
    The first step is multiplying the probabilities of Jill Stein giving a speech with her individual 
    probabilities of saying the words 'freedom' and 'immigration'. Store this in a variable called p_j_text
    
    The second step is multiplying the probabilities of Gary Johnson giving a speech with his individual 
    probabilities of saying the words 'freedom' and 'immigration'. Store this in a variable called p_g_text
    
    The third step is to add both of these probabilities and you will get P(F,I).

    '''
    # Step 1
    # P(J)
    p_j = 0.5
    
    # P(F/J)
    p_j_f = 0.1
    
    # P(I/J)
    p_j_i = 0.1
    
    p_j_text = p_j * p_j_f * p_j_i
    print(p_j_text)
    
    # Step 2
    # P(G)
    p_g = 0.5
    
    # P(F/G)
    p_g_f = 0.7
    
    # P(I/G)
    p_g_i = 0.2
    
    p_g_text = p_g * p_g_f * p_g_i
    print(p_g_text)

    # Step 3
    p_f_i = p_j_text + p_g_text
    print('Probability of words freedom and immigration being said are: ', format(p_f_i))

    '''
    Instructions:
    Compute P(J|F,I) using the formula P(J|F,I) = (P(J) * P(F|J) * P(I|J)) / P(F,I) and store it in a variable p_j_fi
    '''
    p_j_fi = p_j_text / p_f_i
    print('The probability of Jill Stein saying the words Freedom and Immigration: ', format(p_j_fi))

    '''
    Instructions:
    Compute P(G|F,I) using the formula P(G|F,I) = (P(G) * P(F|G) * P(I|G)) / P(F,I) and store it in a variable p_g_fi
    '''

    p_g_fi = p_g_text / p_f_i
    print('The probability of Gary Johnson saying the words Freedom and Immigration: ', format(p_g_fi))



####################################################################################33
