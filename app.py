from joblib import load
import gradio as gr
from sklearn.pipeline import make_pipeline

# Load the model
model_bayes = load('filename.joblib')

# Prediction function
def make_prediction(user_sentence):
  
  prediction = model_bayes.predict([user_sentence])
  dict = {1: 'Negative', 2: 'Neutral', 3: 'Positive'}
  return dict[prediction[0]]

title = "Sentiment Analysis MyAnimeList Reviews"
description = "<p style='text-align: center'>Identifier si un commentaire dans MyAnimeList est positif, neutre ou négatif.<br/> Permet de connaître rapidement le sentiment globale que dégage un avis sur le site.</p>"
examples = ["I liked this show but now I do not love this since the last season. The animation is terrible and the drawings are awful. I don't recommend this show to anyone.", "This is amazing !"]

app = gr.Interface(fn=make_prediction, title=title, description=description, examples=examples, inputs=gr.TextArea(), outputs='text')

app.launch()