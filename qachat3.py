from dotenv import load_dotenv
load_dotenv()  # loads all the environment variables

import streamlit as st
import os
import google.generativeai as genai  # Assuming this is your custom library
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import spacy

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load gemini pro model and get response
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])


def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=True)  # Check documentation for stream
        return response
    except Exception as e:
        print(f"Error getting response: {e}")
        return "An error occurred. Please try again later."


def extract_keywords_and_embedding(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    # Filter out stop words and punctuation marks
    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct]
    # Assuming the model doesn't support text embedding, use doc.vector for simplicity
    embedding = doc.vector  # Adjust based on model requirements
    return keywords, embedding

def extract_keywords(text):  # This function was missing previously
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())
    # Filter out stop words and punctuation marks
    keywords = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return keywords

def get_topic_classification(text):
    # Define a dictionary mapping keywords/phrases to topics
    topic_rules = {
        "weather": ["weather", "forecast", "temperature", "climate", "meteorology", "rainy", "sunny", "cloudy"],
        "sports": ["sports", "game", "football", "soccer", "basketball", "baseball", "hockey", "tennis", "cricket"],
        "technology": ["technology", "computer", "phone", "gadget", "internet", "software", "hardware", "smartphone", "tablet"],
        "finance": ["finance", "stock", "market", "investment", "economy", "business", "money", "financial", "stock market"],
        "health": ["health", "fitness", "nutrition", "exercise", "wellness", "diet", "healthy", "medical", "well-being"],
        "travel": ["travel", "trip", "vacation", "destination", "tourism", "adventure", "explore", "journey", "holiday"],
        "education": ["education", "school", "learning", "college", "university", "student", "teacher", "classroom", "study"],
        "maths": ["maths", "mathematics", "algebra", "geometry", "calculus", "equation", "problem", "theorem",
          "arithmetic", "trigonometry", "statistics", "probability", "number theory", "linear algebra",
          "differential equations", "graph theory", "logic", "set theory", "+", "-", "%", "/", "*",
          "integral", "derivative", "matrix", "vector", "complex numbers", "functions", "limits",
          "series", "coordinate geometry", "conic sections", "logarithm", "exponent", "polynomials",
          "probability distributions", "permutation", "combination", "mathematical induction",
          "mathematical logic", "mathematical modeling", "numerical analysis", "game theory",
          "mathematical optimization", "number system", "puzzle solving", "mathematical proofs",
          "mathematical reasoning", "mathematical symbols", "algebraic expressions", "geometric shapes",
          "calculus concepts", "mathematical operations", "mathematical functions", "mathematical constants",
          "mathematical techniques", "mathematical methods", "mathematical principles", "mathematical algorithms",
          "mathematical properties", "mathematical theories", "mathematical models", "mathematical structures",
          "mathematical concepts", "mathematical applications", "mathematical problems", "mathematical challenges",
          "mathematical puzzles", "mathematical patterns", "mathematical trends", "mathematical phenomena",
          "mathematical analysis", "mathematical synthesis", "mathematical deduction", "mathematical inference",
          "mathematical abstraction", "mathematical generalization", "mathematical abstraction",
          "mathematical interpretation", "mathematical representation", "mathematical description",
          "mathematical classification", "mathematical properties", "mathematical relationships",
          "mathematical structures", "mathematical operations", "mathematical techniques", "mathematical procedures",
          "mathematical protocols", "mathematical guidelines", "mathematical standards", "mathematical criteria",
          "mathematical norms", "mathematical rules", "mathematical laws", "mathematical principles",
          "mathematical frameworks", "mathematical systems", "mathematical methodologies", "mathematical approaches",
          "mathematical strategies", "mathematical tactics", "mathematical maneuvers", "mathematical maneuvers",
          "mathematical transformations", "mathematical modifications", "mathematical adjustments",
          "mathematical adaptations", "mathematical variations", "mathematical alterations", "mathematical changes",
          "mathematical updates", "mathematical revisions", "mathematical enhancements", "mathematical improvements",
          "mathematical advancements", "mathematical progress", "mathematical evolution", "mathematical development",
          "mathematical innovation", "mathematical invention", "mathematical discovery", "mathematical exploration",
          "mathematical investigation", "mathematical examination", "mathematical study", "mathematical analysis",
          "mathematical evaluation", "mathematical assessment", "mathematical scrutiny", "mathematical inspection",
          "mathematical observation", "mathematical review", "mathematical audit", "mathematical inquiry",
          "mathematical research", "mathematical inquiry", "mathematical exploration", "mathematical inquiry",
          "mathematical exploration", "mathematical quest", "mathematical search", "mathematical pursuit",
          "mathematical chase", "mathematical hunt", "mathematical journey", "mathematical adventure",
          "mathematical trek", "mathematical voyage", "mathematical expedition", "mathematical odyssey",
          "mathematical safari", "mathematical survey", "mathematical probe", "mathematical delve",
          "mathematical plunge", "mathematical dive", "mathematical leap", "mathematical bound",
          "mathematical jump", "mathematical hop", "mathematical skip", "mathematical bounce",
          "mathematical sprint", "mathematical dash", "mathematical race", "mathematical sprint",
          "mathematical rush", "mathematical charge", "mathematical storm", "mathematical attack",
          "mathematical assault", "mathematical onslaught", "mathematical blitz", "mathematical barrage",
          "mathematical salvo", "mathematical fusillade", "mathematical volley", "mathematical shower",
          "mathematical deluge", "mathematical flood", "mathematical torrent", "mathematical downpour",
          "mathematical stream", "mathematical current", "mathematical tide", "mathematical wave",
          "mathematical surge", "mathematical swell", "mathematical heave", "mathematical bulge",
          "mathematical upsurge", "mathematical uprising", "mathematical uprising", "mathematical breakthrough",
          "mathematical advance", "mathematical leap", "mathematical step", "mathematical stride",
          "mathematical jump", "mathematical bound", "mathematical hop", "mathematical skip",
          "mathematical walk", "mathematical stroll", "mathematical saunter", "mathematical amble",
          "mathematical ramble", "mathematical hike", "mathematical trek", "mathematical tour",
          "mathematical expedition", "mathematical odyssey", "mathematical journey", "mathematical voyage",
          "mathematical pilgrimage", "mathematical excursion", "mathematical outing", "mathematical jaunt",
          "mathematical junket", "mathematical adventure", "mathematical quest", "mathematical pursuit",
          "mathematical chase", "mathematical hunt", "mathematical search", "mathematical exploration",
          "mathematical investigation", "mathematical study", "mathematical"],

        "physics": ["physics", "force", "energy", "motion", "gravity", "electromagnetism", "quantum", "mechanics",
            "thermodynamics", "optics", "kinematics", "relativity", "acoustics", "fluid dynamics", "nuclear physics",
            "particle physics", "solid-state physics", "astrophysics", "cosmology", "quantum field theory",
            "string theory", "general relativity", "special relativity", "classical mechanics", "quantum electrodynamics",
            "quantum chromodynamics", "statistical mechanics", "atomic physics", "condensed matter physics",
            "optical physics", "plasma physics", "molecular physics", "biophysics", "fluid mechanics",
            "quantum gravity", "classical field theory", "particle accelerators", "superconductivity",
            "quantum optics", "atomic spectroscopy", "nuclear magnetic resonance", "particle accelerators",
            "gravitational waves", "black holes", "quantum computing", "high-energy physics", "low-energy physics",
            "ultrafast phenomena", "lasers", "optoelectronics", "thermoelectrics", "nanophysics", "quantum biology",
            "physical constants", "physical laws", "physical principles", "physical quantities", "physical units",
            "physical phenomena", "physical theories", "physical models", "physical methods", "physical approaches",
            "physical strategies", "physical tactics", "physical maneuvers", "physical maneuvers", "physical operations",
            "physical transformations", "physical modifications", "physical adjustments", "physical adaptations",
            "physical variations", "physical alterations", "physical changes", "physical updates", "physical revisions",
            "physical enhancements", "physical improvements", "physical advancements", "physical progress",
            "physical evolution", "physical development", "physical innovation", "physical invention",
            "physical discovery", "physical exploration", "physical investigation", "physical examination",
            "physical study", "physical analysis", "physical evaluation", "physical assessment", "physical scrutiny",
            "physical inspection", "physical observation", "physical review", "physical audit", "physical inquiry",
            "physical research", "physical inquiry", "physical exploration", "physical inquiry", "physical exploration",
            "physical quest", "physical search", "physical pursuit", "physical chase", "physical hunt",
            "physical journey", "physical adventure", "physical trek", "physical voyage", "physical expedition",
            "physical odyssey", "physical safari", "physical survey", "physical probe", "physical delve",
            "physical plunge", "physical dive", "physical leap", "physical bound", "physical jump",
            "physical hop", "physical skip", "physical bounce", "physical sprint", "physical dash",
            "physical race", "physical sprint", "physical rush", "physical charge", "physical storm",
            "physical attack", "physical assault", "physical onslaught", "physical blitz", "physical barrage",
            "physical salvo", "physical fusillade", "physical volley", "physical shower", "physical deluge",
            "physical flood", "physical torrent", "physical downpour", "physical stream", "physical current",
            "physical tide", "physical wave", "physical surge", "physical swell", "physical heave",
            "physical bulge", "physical upsurge", "physical uprising", "physical uprising", "physical breakthrough",
            "physical advance", "physical leap", "physical step", "physical stride", "physical jump",
            "physical bound", "physical hop", "physical skip", "physical walk", "physical stroll",
            "physical saunter", "physical amble", "physical ramble", "physical hike", "physical trek",
            "physical tour", "physical expedition", "physical odyssey", "physical journey", "physical voyage",
            "physical pilgrimage", "physical excursion", "physical outing", "physical jaunt", "physical junket",
            "physical adventure", "physical quest", "physical pursuit", "physical chase", "physical hunt",
            "physical search", "physical exploration", "physical investigation", "physical study"],

        "chemistry": ["chemistry", "chemical", "element", "compound", "molecule", "reaction", "atom", "bond",
              "stoichiometry", "organic chemistry", "inorganic chemistry", "physical chemistry", "biochemistry",
              "analytical chemistry", "polymer chemistry", "environmental chemistry", "medicinal chemistry",
              "chemical engineering", "chemical equations", "chemical properties", "chemical reactions",
              "chemical bonds", "chemical compounds", "chemical structures", "chemical formulas",
              "chemical compositions", "chemical transformations", "chemical changes", "chemical modifications",
              "chemical adaptations", "chemical variations", "chemical alterations", "chemical updates",
              "chemical revisions", "chemical enhancements", "chemical improvements", "chemical advancements",
              "chemical progress", "chemical evolution", "chemical development", "chemical innovation",
              "chemical invention", "chemical discovery", "chemical exploration", "chemical investigation",
              "chemical examination", "chemical study", "chemical analysis", "chemical evaluation",
              "chemical assessment", "chemical scrutiny", "chemical inspection", "chemical observation",
              "chemical review", "chemical audit", "chemical inquiry", "chemical research", "chemical inquiry",
              "chemical exploration", "chemical inquiry", "chemical exploration", "chemical quest",
              "chemical search", "chemical pursuit", "chemical chase", "chemical hunt", "chemical journey",
              "chemical adventure", "chemical trek", "chemical voyage", "chemical expedition",
              "chemical odyssey", "chemical safari", "chemical survey", "chemical probe", "chemical delve",
              "chemical plunge", "chemical dive", "chemical leap", "chemical bound", "chemical jump",
              "chemical hop", "chemical skip", "chemical bounce", "chemical sprint", "chemical dash",
              "chemical race", "chemical sprint", "chemical rush", "chemical charge", "chemical storm",
              "chemical attack", "chemical assault", "chemical onslaught", "chemical blitz", "chemical barrage",
              "chemical salvo", "chemical fusillade", "chemical volley", "chemical shower", "chemical deluge",
              "chemical flood", "chemical torrent", "chemical downpour", "chemical stream", "chemical current",
              "chemical tide", "chemical wave", "chemical surge", "chemical swell", "chemical heave",
              "chemical bulge", "chemical upsurge", "chemical uprising", "chemical uprising", "chemical breakthrough",
              "chemical advance", "chemical leap", "chemical step", "chemical stride", "chemical jump",
              "chemical bound", "chemical hop", "chemical skip", "chemical walk", "chemical stroll",
              "chemical saunter", "chemical amble", "chemical ramble", "chemical hike", "chemical trek",
              "chemical tour", "chemical expedition", "chemical odyssey", "chemical journey", "chemical voyage",
              "chemical pilgrimage", "chemical excursion", "chemical outing", "chemical jaunt", "chemical junket",
              "chemical adventure", "chemical quest", "chemical pursuit", "chemical chase", "chemical hunt",
              "chemical search", "chemical exploration", "chemical investigation", "chemical study"],

        # Add more topics and related keywords/phrases as needed
    }
 

    # Find the first matching topic based on keywords
    for topic, keywords in topic_rules.items():
        if any(keyword in text.lower() for keyword in keywords):
            return topic

    # Return a default topic if no match is found
    return "General"

def check_similarity(question, prev_response):
    question_keywords, question_embedding = extract_keywords_and_embedding(question)
    prev_response_keywords, prev_response_embedding = extract_keywords_and_embedding(prev_response)
    # Jaccard Similarity using keywords (alternative to cosine similarity)
    intersection = len(set(question_keywords) & set(prev_response_keywords))
    union = len(set(question_keywords) | set(prev_response_keywords))
    if union == 0:
        return 0  # Avoid division by zero
    jaccard_similarity = intersection / union
     # 2. Topic similarity (using rule-based classification)
    current_topic = get_topic_classification(question)
    prev_topic = get_topic_classification(prev_response)

    # Combine similarity scores (adjust weights as needed)
    combined_similarity = 0.3 * jaccard_similarity + 0.7 * (current_topic == prev_topic)

    return combined_similarity >= 0.5 


# initializing streamlit app
st.set_page_config(page_title="Conversational QnA")
st.header("Conversational QnA Chatbot")
st.subheader("Gemini LLM Application")

# initializing session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
    st.session_state['prev_response'] = None

user_input = st.text_input("Input:", key="input")  # user input or question
submit = st.button("Ask the question")

if submit and user_input:
    if not st.session_state['prev_response']:
        # If there's no previous response, treat the input as the first question
        response = get_gemini_response(user_input)
        st.session_state['prev_response'] = user_input
    else:
        prev_keywords = extract_keywords(st.session_state['prev_response'])
        current_keywords = extract_keywords(user_input)
        if any(keyword in prev_keywords for keyword in current_keywords):
            similarity = check_similarity(user_input, st.session_state['prev_response'])
            if similarity >= 0.5:  # Adjust context threshold as needed
                response = get_gemini_response(user_input)
                st.session_state['prev_response'] = user_input
            else:
                response = "This question seems unrelated to the previous conversation. Please rephrase or start a new topic."
        else:
            response = "This question does not seem to follow the previous conversation."

    st.session_state['chat_history'].append(("You", user_input))
    st.subheader("The Response is ")


    complete_response = ""
    for chunk in response:
        if hasattr(chunk, 'text'):  # Check if chunk has a 'text' attribute
            complete_response += chunk.text  # Access text attribute if available
        else:
            complete_response += chunk  # Add string directly otherwise

    st.write(complete_response)
    st.session_state['chat_history'].append(("Bot", complete_response))

# Displaying chat history with basic sanitization (replace `<` and `>` with escape characters)
st.subheader("The Chat History is ")
for role, text in st.session_state['chat_history']:
    escaped_text = text.replace("<", "&lt;").replace(">", "&gt;")
    st.write(f"{role}: {escaped_text}", unsafe_allow_html=False)