"""
Contruction du modele de 
"""
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def display_predicted_topics(model, feature_names, num_top_words,topic_names=None):
    """
    Cette fonction permet d'afficher les mots clés
    associer au sujet 
    """

    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i] \
             for i in topic.argsort()[:-num_top_words - 1:-1]]))
        
def contruction_model(df):
    """
    Fonction pour la contruction et la savugarde des modeles
    """

    vectorizer= TfidfVectorizer(ngram_range=(1,1),max_df=0.8,min_df=0.2)
    df = vectorizer.fit_transform(df.text_cleaned)
    matrix_df = pd.DataFrame(df.toarray(),columns=vectorizer.get_feature_names())
    matrix_df.index = data.index
    model_nmf = NMF(15)
    #theme_sujet = model_nmf.fit_transform(matrix_df)
    #display_predicted_topics(model_nmf, vectorizer.get_feature_names(), 10)
    with open('/home/toffe/PROJET_FE/Topic_Modeling/Topic_Modeling/model/model.pkl','wb') as file:
        pickle.dump(model_nmf, file)
    with open('/home/toffe/PROJET_FE/Topic_Modeling/Topic_Modeling/model/vectoriser.pkl','wb') as file:
        pickle.dump(vectorizer, file)
    print("tache bien effectue")

data = pd.read_csv("/home/toffe/PROJET_FE/Topic_Modeling/Topic_Modeling/data/dataset_negative.csv")
contruction_model(data)
