# Career Buddy — Career Guidance Bot

An AI-powered chatbot that guides students on international education — diplomas, degrees, masters, student visas, costs, and study options in the UK, USA, Canada, Australia, Germany, and New Zealand.

Built with **Flask**, **scikit-learn**, and **NLTK**, the bot uses keyword/pattern matching combined with a trained ML model to answer student queries instantly through a simple web chat interface.

---

## Features

- Real-time chat interface (no page reload — AJAX-based)
- 36+ topics covered — qualification levels (Diploma, HND, Top-up Degree, Bachelor's, Master's, PhD), country guides, visas, costs, scholarships, IELTS, accommodation, and more
- Two-layer answer engine: fast keyword matching + ML model (TF-IDF + Logistic Regression) fallback for fuzzy/unseen questions
- Basic profanity filter using NLTK tokenization
- Easy-to-edit knowledge base (`intents.json`) — add new topics without touching code
- Clean, responsive chat UI built with Bootstrap

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| NLP / Text Processing | NLTK (`word_tokenize`) |
| Machine Learning | scikit-learn (TF-IDF Vectorizer + Logistic Regression) |
| Model Storage | Pickle |
| Knowledge Base | JSON |
| Frontend | HTML, CSS, Bootstrap 4, jQuery, Font Awesome |

---

## How It Works

```
User Message
     │
     ▼
1️⃣  Keyword/Pattern Matching → exact match, contains-match, word-overlap scoring
     │  (no match)
     ▼
2️⃣  ML Model Prediction → TF-IDF + Logistic Regression (confidence ≥ 0.12)
     │  (no confident match)
     ▼
3️⃣  Fallback Response → "I'm not sure I understood that..."
```

Every word is also checked against a blocked-words list before processing, to filter out inappropriate input.

---

## Project Structure

```
career_bot/
├── app.py                # Flask server + chatbot logic
├── train_model.py        # Trains the ML model from intents.json
├── intents.json           # Knowledge base (questions + answers)
├── bot_model.pkl          # Pre-trained ML model
├── blocked.txt             # Profanity filter word list
├── requirements.txt        # Python dependencies
├── static/
│   ├── style.css
│   ├── logo.jpg
│   ├── robo.png
│   └── user.png
└── templates/
    └── index.html         # Chat UI
```

---

## Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/career_guidance_bot.git
cd career_guidance_bot/career_bot

# Install dependencies
pip install -r requirements.txt
```

### (Optional) Re-train the model
Run this only if you've edited `intents.json`:
```bash
python train_model.py
```

### Run the app
```bash
python app.py
```

Then open your browser at:
```
http://localhost:5000
```

---

## Adding New Topics

To teach the bot a new topic, just add a new entry to `intents.json`:

```json
{
  "tag": "new_topic",
  "patterns": ["example question 1", "example question 2"],
  "responses": ["Your answer here"]
}
```

No code changes needed. Re-run `train_model.py` afterward so the ML fallback also learns the new patterns.

---

## Screenshots

<img width="1918" height="962" alt="image" src="https://github.com/user-attachments/assets/9f541d48-3e2c-4099-b9ef-96c91467ff9c" />
_

---

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## License

This project is open source. Add your preferred license (e.g. MIT) here.

---

## Acknowledgements

Built as a student-focused tool to simplify access to study-abroad and career pathway information.
