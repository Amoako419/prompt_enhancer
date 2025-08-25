# 🚀 Prompt Enhancer Chat

A lightweight web app that lets you **chat with Google Gemini** and **auto-enhance prompts** before sending them.

Features
--------
- Chat-style interface (User ↔ Assistant)  
- One-click **prompt enhancement** (✨ button)  
- Retry / Cancel / Replace flow for enhanced prompts  
- FastAPI backend, React + Vite frontend  
- Ready to deploy anywhere (Docker, Render, Vercel, etc.)

Tech Stack
----------
| Layer    | Technology |
|----------|------------|
| Backend  | FastAPI + Google Gemini 2.0 Flash |
| Frontend | React 18 + Vite + Axios + Lucide Icons |
| Language | Python 3.10+ / Node 18+ |

Prerequisites
-------------
- **Python 3.10+**  
- **Node 18+** (ships with `npm`)  
- A **Google Gemini API key** (get one from [Google AI Studio](https://makersuite.google.com/app/apikey))

Project Structure
-----------------
```
prompt-enhancer/
├── backend/
│   ├── main.py
│   └── .env
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── ...
│   ├── package.json
│   └── vite.config.js
└── README.md
```

Getting Started
---------------

### 1. Clone / download the repo
```bash
git clone https://github.com/amoako419/prompt-enhancer.git
cd prompt-enhancer
```

### 2. Backend Setup

#### 2.1 Create virtual environment
```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

#### 2.2 Install dependencies
```bash
pip install -r requirements.txt
# or manually
pip install fastapi uvicorn google-generativeai python-dotenv
```

#### 2.3 Environment variables
Create `backend/.env`:
```
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

#### 2.4 Run
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend is live on http://localhost:8000  
Docs: http://localhost:8000/docs

### 3. Frontend Setup

#### 3.1 Install deps
```bash
cd ../frontend
npm install
# or: pnpm install / yarn install
```

#### 3.2 Start dev server
```bash
npm run dev
```
Frontend opens at http://localhost:5173

Usage
-----
1. Type a message in the chat box.  
2. **Send** → message is sent to Gemini and the assistant replies.  
3. **✨** (wand icon) → opens a dialog with an improved prompt.  
   - **Replace** – puts the enhanced prompt back into the input.  
   - **Retry** – regenerates the enhanced prompt.  
   - **Cancel** – closes the dialog.

API Endpoints
-------------
| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| POST   | `/enhance` | `{"prompt": "raw text"}` | `{"enhanced_prompt": "..."}` |
| POST   | `/chat`    | `{"history": [{"role":"user", "text":"..."}]}` | `{"reply": "..."}` |

Environment Variables
---------------------
| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required) |

Deployment Tips
---------------
- **Backend**: Render, Railway, Fly.io, or any VPS (`uvicorn main:app --host 0.0.0.0 --port $PORT`)  
- **Frontend**: Vercel, Netlify, or build static files (`npm run build`) and serve from backend (`/static` mount).  
- Remember to set `GEMINI_API_KEY` in the host’s environment variables.

Contributing
------------
Pull requests welcome! Please open an issue first for large changes.

License
-------
MIT © 2024

