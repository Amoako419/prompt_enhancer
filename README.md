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



---

## 🆕 Latest Features & Updates

### 📊 MCP Data Analysis Module

*Advanced data analysis capabilities with visual insights*

#### Core Features:

- **File Upload**: Support for CSV data files with drag-and-drop interface
- **Multiple Analysis Types**:
  - Statistical Analysis (descriptive statistics, correlations)
  - Data Visualizations (histograms, scatter plots, heatmaps, box plots)
  - Hypothesis Testing (t-tests, chi-square tests, ANOVA)
  - Machine Learning (clustering analysis with K-means)

#### Enhanced User Experience:

##### 🖼️ Interactive Visualizations

- **Click-to-Expand Images**: All generated charts and visualizations are clickable
- **Full-Screen Modal View**: Images open in responsive modal dialogs
- **Download Functionality**: Save visualizations as PNG files
- **Mobile-Responsive**: Touch-friendly interactions on all devices

##### 📱 Collapsible Content Sections

- **Summary Sections**: All analysis summaries can be collapsed/expanded
  - Statistical Analysis summary
  - Visualization summary
  - Hypothesis Testing summary
  - ML Model summary
- **Smooth Animations**: CSS transitions for expand/collapse actions
- **Consistent UI**: Unified design across all collapsible components

##### 🎨 Enhanced Styling

- **Dark Theme**: Professional dark mode interface
- **Hover Effects**: Interactive feedback on all clickable elements
- **Icon Integration**: Lucide React icons for improved visual hierarchy
- **Responsive Grid Layouts**: Optimized for all screen sizes

#### Technical Implementation:

##### Backend Enhancements:

- **Image Generation**: Server-side chart creation using Matplotlib/Seaborn
- **Base64 Encoding**: Efficient image data transfer to frontend
- **Standardized API**: Consistent response format across all endpoints
- **ML Visualizations**: Four types of cluster analysis charts:
  - Cluster scatter plots
  - Distribution histograms
  - Feature box plots
  - Cluster centers heatmaps

##### Frontend Architecture:

- **Component Structure**:

```text
MCPDataAnalysis.jsx (Main container)
├── MCPAnalysisTypes.jsx (Analysis components)
│   ├── StatisticalAnalysis (with collapsible summary)
│   ├── Visualizations (with click-to-expand)
│   ├── StatisticalTests (with collapsible summary)
│   └── MLModelResults (with interactive charts)
└── ImageModal (Full-screen image viewer)
```

##### State Management:

- **Modal States**: Click-to-expand functionality for images
- **Collapse States**: Individual toggle states for each summary section
- **Responsive Interactions**: Touch and mouse event handling

#### API Endpoints:

| Endpoint | Method | Purpose | Features |
|----------|--------|---------|----------|
| `/mcp/visualization` | POST | Generate data visualizations | Returns charts as base64 images |
| `/mcp/hypothesis-testing` | POST | Statistical hypothesis tests | Comprehensive test results |
| `/mcp/machine-learning` | POST | K-means clustering analysis | Visual cluster analysis |

#### CSS Architecture:

- **Modular Styling**: Separate CSS files for each component
- **Animation System**: Smooth transitions and hover effects
- **Modal System**: Overlay design with backdrop blur
- **Responsive Breakpoints**: Mobile-first responsive design

#### User Interaction Flow:

1. **Data Upload**: Drag-and-drop CSV files or click to browse
2. **Analysis Selection**: Choose from 4 analysis types
3. **Interactive Results**:
   - Click images to view full-screen
   - Toggle summaries to save screen space
   - Download visualizations for reports
4. **Mobile Experience**: Touch-optimized for all devices

#### Performance Optimizations:

- **Lazy Loading**: Components render only when needed
- **Image Optimization**: Efficient base64 encoding
- **CSS Animations**: Hardware-accelerated transitions
- **Responsive Images**: Optimized sizing for different viewports

### 🛠️ Technical Improvements:

#### Static Layout Enhancement:

- **Sidebar Behavior**: Made sidebar scroll naturally with page content
- **Header Positioning**: Removed fixed positioning for better document flow
- **Container Layout**: Optimized flex layouts for natural scrolling

#### Icon System:

- **Expanded Icon Library**: Added ChevronDown, ChevronUp, ZoomIn, Download icons
- **Consistent Usage**: Standardized icon sizing and styling across components

#### Development Experience:

- **Modular Architecture**: Clean separation of concerns
- **Reusable Components**: Standardized modal and collapsible patterns
- **Consistent Styling**: Unified design system across all components

---

### 🚀 Getting Started with New Features:

1. **Navigate to MCP Data Analysis** from the main application
2. **Upload a CSV file** using the drag-and-drop interface
3. **Select an analysis type** (Statistical, Visualization, Testing, or ML)
4. **Execute analysis** and explore interactive results
5. **Click on images** to view them full-screen
6. **Toggle summaries** to manage screen space
7. **Download visualizations** for external use

### 📱 Mobile Support:

All new features are fully responsive and touch-optimized for mobile devices, including:

- Touch-friendly click-to-expand images
- Optimized modal sizing for small screens
- Responsive grid layouts
- Touch gesture support for all interactions


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