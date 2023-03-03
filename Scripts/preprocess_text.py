"""
Prepocess du dataframe
"""

# importation des librairies
import contractions
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# initialisation de certaines varibles qui seront utiles
tokenizer = RegexpTokenizer(r'\w+')
lemmentizer = WordNetLemmatizer()
negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"

#definition des fonctions de pritrement 
def tokenizer_text(text):
    """fonction pour tokeniser ces textes
    en entré nous avons un text:str
    en sortie aussi un text:str
    """
    text_traiter =" ".join(tokenizer.tokenize(text))
    return text_traiter

def lemmentizer_text(text):
    """
    Fonction pour lemmentiser les mot du textes
    en entré un text : str , en sortie un text :str
    """
    token_tag = nltk.pos_tag(nltk.word_tokenize(text))
    lemmentizer_text_list = list()
    for word,tag in token_tag:
        if tag.startswith('J'):
            lemmentizer_text_list.append(lemmentizer.lemmatize(word,
                                        'a'))#lemmentise les adjectifs
        elif tag.startswith('V'):
            lemmentizer_text_list.append(lemmentizer.lemmatize(word,
                                        'v'))# lemmentisation des verbs
        elif tag.startswith('N'):
            lemmentizer_text_list.append(lemmentizer.lemmatize(word,
                                        'n'))# lemmentise les noms
        elif tag.startswith('R'):
            lemmentizer_text_list.append(lemmentizer.lemmatize(word,
                                        'r'))#lemmentise les verbs
        else:
            lemmentizer_text_list.append(
                lemmentizer.lemmatize(word)
            )
    return " ".join(lemmentizer_text_list)

def upper_to_lower(text):
    """
    Cette fonction transforme tout les mots d'un
    text en minuscule
    prend comme entrer un text
    retourne un text en miniscule

    """
    text_lower = [word.lower()  for word in text.split()]
    return " ".join(text_lower)

def fix_contractions(text):
    """
    Cette fonction permet de fixer les contractions 
    elle prend entrée un text:str 
    et en sortie elle retourne un text : str
    """
    return contractions.fix(text)

def token_negatif(text):
    """
    Cette fonction permet d'avoir les tokens negatif 
    elle prend en entré un text : str 
    en sortie elle retourne un tet : str 
    """
    tokens = text.split()
    negative_idx = [i+1 for i in range(len(tokens)-1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx<len(tokens):
            tokens[idx] = negative_prefix+tokens[idx]
    tokens = [token for i, token in enumerate(tokens) if i + 1 not in negative_idx]
    return " ".join(tokens)

def remove_stopwords(text):
    """
    elle permet de supprimer les stopswords *
    en etrée un text et en sortie un text
    """
    english_stopwords = stopwords.words("english") + list(STOP_WORDS) + ["tell", "restaurant"]

    return " ".join([word for word in text.split() if word not in english_stopwords])

def preprocessing(text):
    """
    Cette fonction recapitule tout les autres etapes du
    traitement en appliquant les fonctions 
    elle prend un text en entrée et en sortie un text
    
    """
    # tokenizer
    text = tokenizer_text(text)
    # lemmentizer text
    text = lemmentizer_text(text)
    #rendre en miniscule 
    text = upper_to_lower(text)
    #suppression des contactions 
    text =  fix_contractions(text)
    # avoir les tokens negatif 
    text = token_negatif(text)
    #suppression des stopwords
    text = remove_stopwords(text)
    return text

CHEMIN_DATASET = "../data/dataset.csv"
dataset_df = pd.read_csv(CHEMIN_DATASET)
dataset_df["length"] = dataset_df["text"].apply(lambda x: len(x.split()))

# Enregistrer Cleaned Dataset
dataset_df["text_cleaned"] = dataset_df["text"].apply(preprocessing)
dataset_df.to_csv("../data/dataset_nettoyer.csv", index=False)

# Extraction du jeu de données d'avis négatifs
dataset_negative_df = dataset_df[dataset_df.stars < 3]
dataset_negative_df.to_csv("../data/dataset_negative.csv", index=False)
