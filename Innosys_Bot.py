#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import re
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
user_input_saved = st.session_state.get("user_input_saved", [])
#global q_index


messages = [
    {"role": "system", "content": ""},
]

#Erstelle einen Datframe mit Inhalten und den dazugehörigen Embeddings
df_try =pd.read_csv('df_InnoSys_Angebote_lang.csv')
all_embeddings = np.load('embeddings_InnoSys_Angebote_lang.npy', allow_pickle=True)
df_try['ada_v2_embedding'] = all_embeddings

colx_head, coly_head, colz_head = st.columns([1 ,4,1])
colx_top, coly_top, colz_top = st.columns([1 ,4,1])
colx_bot, coly_bot, colz_bot = st.columns([1 ,4,1]) 

ai_response = ""
ai_question = ""

#old_questions = ["Was hat Sie dazu bewegt, Ihr Unternehmen zu gründen?",    "Welche Produkte oder Dienstleistungen bietet Ihr Unternehmen an?",    "Welche Zielgruppe sprechen Sie mit Ihrem Angebot an?",    "Was unterscheidet Ihr Unternehmen von anderen in der Branche?",    "Welche Ziele hat Ihr Unternehmen für die nächsten fünf Jahre?"]   #h Ihr Unternehmen seit der Gründung entwickelt?",    "Welche Herausforderungen mussten Sie auf dem Weg zum Erfolg meistern?",    "Wie sieht Ihr Geschäftsmodell aus?",    "Wie hoch ist der Umsatz Ihres Unternehmens?",    "Wie viele Mitarbeiter hat Ihr Unternehmen?",    "Wie wichtig ist Ihnen eine gute Unternehmenskultur?",    "Welche Marketingstrategien nutzen Sie, um Ihr Unternehmen bekannt zu machen?",    "Wie gehen Sie mit Kundenbeschwerden um?",    "Welche Rolle spielen Innovationen in Ihrem Unternehmen?",    "Wie wichtig ist Ihnen Nachhaltigkeit in Ihrem Unternehmen?",    "Wie sichern Sie die Qualität Ihrer Produkte oder Dienstleistungen?",    "Wie wichtig sind Ihnen Kundenfeedback und Kundenbindung?",    "Welche Rolle spielen Social Media und Online-Marketing in Ihrem Unternehmen?",    "Welche Vision haben Sie für Ihr Unternehmen?",    "Wie planen Sie, Ihr Unternehmen in Zukunft zu erweitern?"]
questions = [
    "Welche aktuellen Herausforderungen sieht das Unternehmen, und wo könnten externe Unterstützung oder zusätzliche Ressourcen von Vorteil sein?",
    "Welche strategischen Ziele verfolgt das Unternehmen derzeit, und auf welche Weise möchten sie diese Ziele erreichen?",
    "Gibt es spezifische Fähigkeiten oder Fachkenntnisse, die das Unternehmen aktuell benötigt, und wie könntest du deine Fähigkeiten einbringen, um dabei zu helfen?",
]
statistics_system_prompt = '''Du bist ein Guide für Angebote vom Projekt InnoSys Nordwest. Das Projekt sitzt an der Hochschule Emden/Leer. Du sollst aus einem langen Text an Angeboten, die passenden Angebote extrahieren. Rede ausschließlich in deutscher Sprache. Der Nutzer gibt dir eine Beschreibung seiner Situation, dann gehst du die Angebote Eintrag für Eintrag durch. Einträge sind durch ___________________ getrennt. Ein Angebot hat folgendes Format:
|Titel|
Beschreibung:...
Typ:...
Standort bzw. Wo:...
Wenn du meinst, dass ein Angebot zu der Anfrage des Nutzers passt, extrahiere den Titel und ausschließlich den Titel. Wenn du alle Einträge durchgegangen bist, präsentiere dem Nutzer, die Titel, die du extrahiert hast in Stichpunkten. Wenn kein Angebot zu der Anfrage passen sollte, versuche die Frage nicht weiter zu beantworten. Hier sind die Angebote: '''

summary_answer_prompt = "Du bekommst mehrere Antworten zu sehen. In diesen Antworten werden Angebote aufgezählt. Bitte nutze die Antworten, die Ergebnisse liefern, um eine Stichpunktliste der Titel und ihrer Beschreibungen zu erstellen. Titel haben folgendes Format |...|. Die Stichpunktliste soll dieses Format haben: |Titel| - Beschreibung. Halte dabei folgende Frage im Kopf: "

system_prompt_new = ''' Du bist ein Guide für die Angebote von InnoSys Nordwest. Du sollst basierend auf der Beschreibung des Nutzers und der Angebote, die er präsentiert, herausfinden von welchen Angeboten der Nutzer am meisten profitiert. Hier sind weitere Eigenschaften von dir: 
Verfügt über weitreichende Technologie- und Branchenkenntnisse.
Ehrlich, wenn eine Frage nicht beantwortet werden kann, und ggf. Weiterleitung an Experten.
Antworte ausschließlich auf Basis der Angebote, die der Nutzer in seiner Anfrage als Referenz gibt.
'''


statements = ["Ich habe mein Unternehmen gegründet, weil: ", "Mein Unternehmen bietet folgende Produkte oder Dienstleistungen an: ", "Ich spreche mit meinem Angebot folgende Zielgruppe an: ", "Mein Unternehmen unterscheidet sich von anderen in der Branche durch: ", "Meine Ziele für die nächsten fünf Jahre sind: "]
welcome_msg = "Hallo! Ich bin der Innosys Bot, der Ihnen dabei helfen wird, die passendsten Angebote für Ihr Unternehmen und die Herausforderungen der Branche zu finden. Beantworten Sie einfach meine Fragen. Desto genauer Sie antworten, desto besser werden meine Vorschäge sein. "
result_msg = "Danke für Ihre Zeit! Hier sind Ihre Antworten: \n"

def generate_summary(query, answers):
    messages = [
        {"role": "system", "content": f"{summary_answer_prompt} {query}"},
        {"role": "user", "content": answers}
    ]

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    ai_response_final = chat.choices[0].message.content

    return ai_response_final

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

def narrowing_it_down(query, text):
    print("BEschreibung: " + query)
    print("Referenz: " + text)
    #answer_list = []
    #for reference in references:
        #tokens = len(tokenizer.encode(reference))
        #print(tokens)
    initial_prompt = "Du bist ein hilfreicher Assistent, der Texte basierend auf einer Beschreibung zusammenfasst."
    add_to_user_query = "Welche Angebote passen zu mir, basierend auf der folgenden Beschreibung: "
    statistics_user_prompt = '''Ich gebe dir eine Beschreibung meiner Situation. Basierend auf dieser Beschreibung gehst, du den Text "Angebote" Angebot für Angebot durch. Angebote sind durch ___________________ getrennt. Ein Angebot hat folgendes Format:
|Titel|
Beschreibung:...
Typ:...
Standort bzw. Wo:...
Wenn du meinst, dass ein Angebot zu meiner Beschreibung passt, extrahiere den Titel. Wenn du alle Angebote durchgegangen bist, präsentiere mir die Angebote, die du extrahiert hast in Stichpunkten, mit einer Begründung warum das Angebot zu mir passt. Wenn kein Angebot zu mir passen sollte, sag es mir. Hier ist meine Beschreibung: '''

    disclaimer = "WICHTIG: Beziehe dich ausschließlich auf die Angebote bei der Beantwortung der Fragen."
    messages = [
        {"role": "system", "content": initial_prompt },
        {"role": "user", "content": f"Gib mir basierend auf dieser Beschreibung: {query} Die passenden Angebote aus dem folgenden Text: {text}"}
    ]

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2
    )

    ai_response = chat.choices[0].message.content
    #answer_list.append("Antwort: " + ai_response)
    print("Hier ist die Antwort: ")
    print(ai_response)
    print("\n")
    print("________________")

    #return answer_list
    return ai_response

def find_and_append(values, possible_values):
    final = list()
    for item in values:
        if item in possible_values:
            final.append(item)
    return final

def ask_question(query, angebote):
    #answer_list = []
    #for reference in references:
        #tokens = len(tokenizer.encode(reference))
        #print(tokens)
    initial_prompt = "Du bist ein Guide für die Angebote von InnoSys Nordwest. Du sollst basierend auf den Eingaben des Nutzers, herausfinden von welchen Angeboten der Nutzer am meisten profitiert."
    add_to_user_query = "Welche Angebote passen zu mir, basierend auf der folgenden Beschreibung: "
    statistics_user_prompt = '''Ich gebe dir eine Beschreibung meiner Situation. Basierend auf dieser Beschreibung gehst, du den Text "Angebote" Angebot für Angebot durch. Angebote sind durch ___________________ getrennt. Ein Angebot hat folgendes Format:
|Titel|
Beschreibung:...
Typ:...
Standort bzw. Wo:...
Wenn du meinst, dass ein Angebot zu meiner Beschreibung passt, extrahiere den Titel. Wenn du alle Angebote durchgegangen bist, präsentiere mir die Angebote, die du extrahiert hast in Stichpunkten, mit einer Begründung warum das Angebot zu mir passt. Wenn kein Angebot zu mir passen sollte, sag es mir. Hier ist meine Beschreibung: '''

    disclaimer = "WICHTIG: Beziehe dich ausschließlich auf die Angebote bei der Beantwortung der Fragen."
    messages = [
        {"role": "system", "content": f"{statistics_system_prompt}Referenzen: {angebote} Beziehe dich ausschließlich auf die Referenzen bei der Beantwortung der Fragen. Wenn du Angebote findest schreibe sie in Stichpunkten auf und erwähne immer den Titel indem du ihn in diesem Stil kennzeichnest |...|. Die drei Punkte repräsentieren den jeweiligen Titel."},
        {"role": "user", "content":  query }
	#{"role": "user", "content":  statistics_user_prompt + query + "Hier ist der Text 'Angebote': "+ angebote + disclaimer }
    ]

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0.2
    )

    ai_response = chat.choices[0].message.content
    #answer_list.append("Antwort: " + ai_response)
    print("Hier ist die Antwort: ")
    print(ai_response)
    print("\n")
    print("________________")

    #return answer_list
    #summary_response = generate_summary(query, ai_response)
    #final_response = narrowing_it_down(query, ai_response)
    return ai_response #summary_response

q_index = st.session_state.get("q_index", 1)


st.session_state["q_index"] = q_index
counter = 0 
coly_head.write(welcome_msg)
chat_message_first(questions[0])
#chat_message_style(questions[q_index])
#chat_history.append(questions[0])
if __name__== '__main__':
    #Get the content
    with open('Angebote.txt', 'r', encoding='utf-8') as file:
	    content = file.read()
	    alle_angebote = (str(content))
    possible_values = re.findall(r'\|([^|]+)\|', alle_angebote)
    #st.write(alle_angebote)
    #chat_history.append(welcome_msg + questions[q_index])
    #user_input_list = list()
    user_input = coly_bot.text_input("Eingabe:")
    send_button = coly_bot.button("Send")

    if send_button and q_index == 3:
        user_input_saved.append(user_input)
        user_input_str = ' '.join(user_input_saved)
        chat_history.append(user_input)
        st.session_state["chat_history"] = chat_history
        st.session_state["user_input_saved"] = user_input_saved
        user_input_str = ' '.join(user_input_saved)
        #st.write(generate_answer())
        #result = generate_answer()
        #result = turn_to_statements()
        result = ask_question(user_input_str, alle_angebote)
        values = re.findall(r'\|([^|]+)\|', result)
	#st.write(result_msg)
	#st.write(user_input_str)
	#st.write("Dies sind meine Vorschläge für Sie: ")
        list_of_entries = find_and_append(values, possible_values)
        if len(list_of_entries) == 0:
            print("Ich konnte keine passenden Anngebote finden")
            user_input_list = list()
        else:
            st.write(result_msg + user_input_str + "\n\n Dies sind meine Vorschläge für Sie: \n" + result)
            #final = narrowing_it_down(user_input_str, result)
            st.write("Die finale Antwort \n" + final)
        user_input_list = list()
        #for chat in chat_history:
            #chat_message_style(chat)

    if send_button and q_index <= 2:
        user_input_saved.append(user_input)
        chat_history.append(user_input)
        chat_history.append(questions[q_index])
        
        #Debug.Log(q_index) 
        q_index += 1
	
        st.session_state["q_index"] = q_index  # aktualisierten Wert im Session State speichern
        st.session_state["user_input_saved"] = user_input_saved
        st.session_state["chat_history"] = chat_history
        for chat in chat_history:
            chat_message_style(chat)


  
    if send_button and q_index == 0:
        #st.write("Done!")
        chat_history.insert(0, questions[0])
        #for chat in chat_history:
            #chat_message_style(chat)
    #else:
        
        


#q_index
#chat_history
#df_try
