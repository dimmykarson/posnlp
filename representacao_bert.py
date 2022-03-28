from sentence_transformers import SentenceTransformer
import pandas as pd

df = pd.read_csv("datasets/imdb.csv")
#Selecionando apenas 350 revisões (performance de aula)
df = df[:350]

#Selecionando as revisões
reviews = df['review']

#importando o modelo pré-treinado
#tipos de modelo:
#https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

vetores = model.encode(reviews)
print(vetores.shape)
import pickle as pck
pck.dump(vetores, 
	open("datasets/representacao_bert.bin",
	"wb"))


