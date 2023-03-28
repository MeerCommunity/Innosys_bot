#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import io
import time
import pandas as pd
import openai
import numpy as np
import streamlit as st
import requests
import webbrowser
from openai.embeddings_utils import cosine_similarity
from num2words import num2words



openai.api_key = st.secrets["api_key"]


#st.session_state["chat_history"] = []
chat_history = st.session_state.get("chat_history", [])
#global q_index


messages = [
    {"role": "system", "content": ""},
]

path = "C:\\Users\\Labor\\"

#Erstelle einen Datframe mit Inhalten und den dazugehörigen Embeddings
df_try =pd.read_csv('df_InnoSys_Angebote_lang.csv')
all_embeddings = np.load('embeddings_InnoSys_Angebote_lang.npy', allow_pickle=True)
df_try['ada_v2_embedding'] = all_embeddings

colx_head, coly_head, colz_head = st.columns([1 ,4,1])
colx_top, coly_top, colz_top = st.columns([1 ,4,1])
colx_bot, coly_bot, colz_bot = st.columns([1 ,4,1]) 

ai_response = ""
ai_question = ""

questions = ["Was hat Sie dazu bewegt, Ihr Unternehmen zu gründen?",    "Welche Produkte oder Dienstleistungen bietet Ihr Unternehmen an?",    "Welche Zielgruppe sprechen Sie mit Ihrem Angebot an?",    "Was unterscheidet Ihr Unternehmen von anderen in der Branche?",    "Welche Ziele hat Ihr Unternehmen für die nächsten fünf Jahre?"]   #h Ihr Unternehmen seit der Gründung entwickelt?",    "Welche Herausforderungen mussten Sie auf dem Weg zum Erfolg meistern?",    "Wie sieht Ihr Geschäftsmodell aus?",    "Wie hoch ist der Umsatz Ihres Unternehmens?",    "Wie viele Mitarbeiter hat Ihr Unternehmen?",    "Wie wichtig ist Ihnen eine gute Unternehmenskultur?",    "Welche Marketingstrategien nutzen Sie, um Ihr Unternehmen bekannt zu machen?",    "Wie gehen Sie mit Kundenbeschwerden um?",    "Welche Rolle spielen Innovationen in Ihrem Unternehmen?",    "Wie wichtig ist Ihnen Nachhaltigkeit in Ihrem Unternehmen?",    "Wie sichern Sie die Qualität Ihrer Produkte oder Dienstleistungen?",    "Wie wichtig sind Ihnen Kundenfeedback und Kundenbindung?",    "Welche Rolle spielen Social Media und Online-Marketing in Ihrem Unternehmen?",    "Welche Vision haben Sie für Ihr Unternehmen?",    "Wie planen Sie, Ihr Unternehmen in Zukunft zu erweitern?"]
statements = ["Ich habe mein Unternehmen gegründet, weil: ", "Mein Unternehmen bietet folgende Produkte oder Dienstleistungen an: ", "Ich spreche mit meinem Angebot folgende Zielgruppe an: ", "Mein Unternehmen unterscheidet sich von anderen in der Branche durch: ", "Meine Ziele für die nächsten fünf Jahre sind: "]
welcome_msg = "Hallo! Ich bin der Innosys Bot, der Ihnen dabei helfen wird, die passendsten Angebote für Ihr Unternehmen und die Herausforderungen der Branche zu finden. Beantworten Sie einfach meine Fragen. Desto genauer Sie antworten, desto besser werden meine Vorschäge sein. "


def chat_message_style(message, is_user=False, col = coly_top): 
    global counter  # Zählervariable deklarieren
    if is_user or counter % 2 == 0: 
        message_format = f""" 
            <div style="display: flex; flex-direction: row-reverse;"> 
                <div style="background-color: #f0f0f0; padding: 5px 10px; border-radius: 10px; margin: 5px 0px;"> 
                    <p style="margin: 0px;">{message}</p> </div> </div> """ 
        #is_user = False 
    else: 
        message_format = f""" 
            <div style="display: flex; flex-direction: row;"> 
                <div style="background-color: #007bff; color: #ffffff; padding: 5px 10px; border-radius: 10px; margin: 5px 0px;"> 
                    <p style="margin: 0px;">{message}</p> </div> </div> """ 
        #is_user = True
    col.empty()
    col.markdown(message_format, unsafe_allow_html=True)
    counter += 1  # Zählervariable erhöhen


def chat_message_first(message, col = coly_top): 
    message_format = f""" 
        <div style="display: flex; flex-direction: row;"> 
            <div style="background-color: #007bff; color: #ffffff; padding: 5px 10px; border-radius: 10px; margin: 5px 0px;"> 
                <p style="margin: 0px;">{message}</p> </div> </div> """ 
        #is_user = True
    col.markdown(message_format, unsafe_allow_html=True)

def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        model="text-embedding-ada-002"
    )
    #Erstelle eine Kopie des Datframes  
    df_question = df.copy()
    
    #Füge eine weitere Spalte hinzu mit Similarity-Score
    df_question["similarities"] = df.ada_v2_embedding.apply(lambda x: cosine_similarity(x, embedding))
    
    #Sortiere den Dataframe basierend auf dem Similarity-Score und zeige die obersten N Einträge
    res = (
        df_question.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    #Greife die obersten N Einträge ab
    return res

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def addToChatHistory(msg, history = chat_history):
    history.append(msg)
    history
    return history

def turn_to_statements(history = chat_history):
    int_insert = 0
    new_history = list()
    for statement in statements:
        new_history.append(statement)
        #new_history.append(history[int_insert])
        #history.insert(int_insert,statement)
        int_insert += 2
    new_history = ["\n".join(history)]
    return new_history

def generate_answer():
    
    user_responses = "Hier sind die Eingaben des Nutzers:" + str(turn_to_statements)
    initial_prompt = "Du bist ein Guide für die Angebote von InnoSys Nordwest. Du sollst basierend auf den Eingaben des Nutzers, herausfinden von welchen Angeboten der Nutzer am meisten profitiert. Schlage dem Nutzer drei Angebote vor und gib eine Begründung."
    #Greife den Eintrag ab, der am meisten Änhlichkeit mit der Frage hat
    res = search_docs(df_try, user_responses, top_n=1)
    #Greife den Inhalt des Eintrages ab
    context= "Hier sind die Angebote, die in Frage kommen: " + res.CONTENT.values
    combined_prompt = initial_prompt + user_responses + str(context)
    #Kombiniere den Prompt mit Baisisprompt, dem Inhalt und der Frage
    
            
    #API-Abfrage
    messages.append(
        {"role": "user", "content": combined_prompt},
    )        
    chat = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", messages=messages
    )
            
    #Greife Inhalt des Resultat der API Abfrage ab
    ai_response = chat.choices[0].message.content
    output = ai_response
    return output


q_index = st.session_state.get("q_index", 1)


st.session_state["q_index"] = q_index
counter = 0 
coly_head.write(welcome_msg)
chat_message_first(questions[0])
#chat_message_style(questions[q_index])
#chat_history.append(questions[0])
if __name__== '__main__':
    #chat_history.append(welcome_msg + questions[q_index])

    user_input = coly_bot.text_input("Eingabe:")
    send_button = coly_bot.button("Send")

    if send_button and q_index == 5:
        chat_history.append(user_input)
        st.session_state["chat_history"] = chat_history
        #st.write(generate_answer())
        result = generate_answer()
        #result = turn_to_statements()
        st.write(result)
        #for chat in chat_history:
            #chat_message_style(chat)

    if send_button and q_index <= 4:
        chat_history.append(user_input)
        chat_history.append(questions[q_index])
        
            
        q_index += 1
        st.session_state["q_index"] = q_index  # aktualisierten Wert im Session State speichern
            
        st.session_state["chat_history"] = chat_history
        for chat in chat_history:
            chat_message_style(chat)


  
    if send_button and q_index == 2:
        #st.write("Done!")
        chat_history.insert(0, questions[0])
        #for chat in chat_history:
            #chat_message_style(chat)
    #else:
        
        


#q_index
#chat_history
#df_try
