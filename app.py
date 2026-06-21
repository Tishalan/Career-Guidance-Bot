from flask import Flask, render_template, request, jsonify
import os
import json
import pickle
import random
import re
from nltk.tokenize import word_tokenize

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCKED_PATH = os.path.join(BASE_DIR, "blocked.txt")
MODEL_PATH = os.path.join(BASE_DIR, "bot_model.pkl")
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")

# Load blocked words
blocked = []
try:
    with open(BLOCKED_PATH, "r", encoding="utf-8", errors="replace") as text_file:
        block = text_file.readlines()
        blocked = [s.strip().lower() for s in block if s.strip()]
except Exception as e:
    print("Warning: Could not load blocked.txt", e)

# Load the trained ML model
ml_model = None
try:
    with open(MODEL_PATH, "rb") as f:
        saved_data = pickle.load(f)
        ml_model = saved_data["model"]
except Exception as e:
    print("Warning: Could not load ML model:", e)

# Load intents data directly from JSON (always fresh)
intents_data = None
try:
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents_data = json.load(f)
except Exception as e:
    print("Error loading intents.json:", e)

# ---- Build a keyword lookup table from intents ----
# Maps keywords/phrases to intent tags for fast matching
keyword_map = {}
if intents_data:
    for intent in intents_data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            # Store the pattern (lowered) -> tag
            keyword_map[pattern.lower().strip()] = tag

def find_intent_by_keywords(user_input):
    """
    Multi-layer keyword matching:
    1. Exact match against all patterns
    2. Check if user input CONTAINS a pattern
    3. Check if a pattern CONTAINS the user input
    4. Partial word overlap scoring
    """
    user_lower = user_input.lower().strip()
    
    # Layer 1: Exact match
    if user_lower in keyword_map:
        return keyword_map[user_lower]
    
    # Layer 2: User input contains a known pattern
    best_match_tag = None
    best_match_len = 0
    for pattern, tag in keyword_map.items():
        if pattern in user_lower and len(pattern) > best_match_len:
            best_match_tag = tag
            best_match_len = len(pattern)
    if best_match_tag and best_match_len >= 3:
        return best_match_tag
    
    # Layer 3: A known pattern contains the user input (for short inputs)
    if len(user_lower) >= 2:
        for pattern, tag in keyword_map.items():
            if user_lower in pattern:
                return tag
    
    # Layer 4: Word overlap scoring
    user_words = set(re.findall(r'\w+', user_lower))
    best_score = 0
    best_tag = None
    for pattern, tag in keyword_map.items():
        pattern_words = set(re.findall(r'\w+', pattern))
        overlap = user_words & pattern_words
        if overlap:
            # Score = number of overlapping words / total unique words
            score = len(overlap) / max(len(user_words | pattern_words), 1)
            if score > best_score:
                best_score = score
                best_tag = tag
    
    if best_score >= 0.4:
        return best_tag
    
    return None

def get_response_for_tag(tag):
    """Get a random response for a given intent tag."""
    if intents_data:
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return None

def fallback_response():
    """Return a helpful fallback response."""
    fallbacks = [
        "I'm not sure I understood that. Try asking about Diploma, HND, Degree, Master's, PhD, or studying in a specific country!",
        "Hmm, I didn't quite catch that. Ask me about international education — Diplomas, Top-up Degrees, Student Visas, or university options!",
        "I'm still learning! You can ask me things like 'What is HND?', 'Study in UK', 'Student visa', or 'What is a Top-up Degree?'",
        "Could you rephrase that? I can help with international education — try asking about qualifications, countries, costs, or scholarships!",
        "I'd love to help! Ask me about Diploma, HND, Bachelor's, Master's, PhD, or studying in UK, USA, Canada, Australia, or Germany!"
    ]
    return random.choice(fallbacks)

def get_bot_answer(user_input):
    if not intents_data:
        return "My knowledge base is not loaded. Please check intents.json."

    # Step 1: Try keyword-based matching (fast and reliable)
    tag = find_intent_by_keywords(user_input)
    if tag:
        response = get_response_for_tag(tag)
        if response:
            return response
    
    # Step 2: Try ML model prediction (handles fuzzy/unseen inputs)
    if ml_model:
        try:
            probs = ml_model.predict_proba([user_input])[0]
            max_prob_index = probs.argmax()
            confidence = probs[max_prob_index]
            
            if confidence >= 0.12:
                predicted_tag = ml_model.classes_[max_prob_index]
                response = get_response_for_tag(predicted_tag)
                if response:
                    return response
        except Exception as e:
            print("ML prediction error:", e)
    
    # Step 3: Fallback
    return fallback_response()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg', '')
    
    # Block check
    try:
        user_words = word_tokenize(userText)
    except:
        user_words = userText.split()
        
    for word in user_words:
        if word.lower() in blocked:
            return "Please avoid swear words!"
         
    response = get_bot_answer(userText)
    
    # Format the response for HTML viewing
    # Convert **bold** to <b>bold</b>
    response_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)
    # Convert newlines to <br> to maintain paragraph structure and lists
    response_html = response_html.replace('\n', '<br>')
    
    return str(response_html)

@app.errorhandler(404)
def not_found(error=None):
    return jsonify({'status': 404, 'message': 'Not Found: ' + request.url}), 404

@app.errorhandler(500)
def internalservererror(error=None):
    return jsonify({'status': 500, 'message': 'Internal Server Error'}), 500

if __name__ == "__main__":
    app.run(port=5000)
