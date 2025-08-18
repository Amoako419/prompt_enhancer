import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { 
  Wand2, RotateCw, Check, X, Send, ArrowLeft, History, 
  Star, Search, Filter, Trash2, Copy, BookOpen, Tag,
  Mic, Plus, Settings
} from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import "../App.css";
import "../styles/PromptEnhancerModern.css";

export default function PromptEnhancer({ onBackToTools }) {
  /* ---------- chat state ---------- */
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loadingReply, setLoadingReply] = useState(false);
  const messagesEndRef = useRef(null);

  /* ---------- enhancement state ---------- */
  const [enhanced, setEnhanced] = useState("");
  const [showDialog, setShowDialog] = useState(false);
  const [loadingEnhance, setLoadingEnhance] = useState(false);

  /* ---------- history & favorites state ---------- */
  const [promptHistory, setPromptHistory] = useState([]);
  const [favorites, setFavorites] = useState([]);
  const [showHistoryPanel, setShowHistoryPanel] = useState(false);
  const [historySearch, setHistorySearch] = useState("");
  const [historyFilter, setHistoryFilter] = useState("all"); // "all", "favorites", "recent"
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [promptToSave, setPromptToSave] = useState({ original: "", enhanced: "", category: "general" });

  /* ---------- load data from localStorage ---------- */
  useEffect(() => {
    const savedHistory = localStorage.getItem('promptHistory');
    const savedFavorites = localStorage.getItem('promptFavorites');
    
    if (savedHistory) {
      setPromptHistory(JSON.parse(savedHistory));
    }
    if (savedFavorites) {
      setFavorites(JSON.parse(savedFavorites));
    }
  }, []);

  /* ---------- save data to localStorage ---------- */
  const saveToStorage = (history, favs) => {
    localStorage.setItem('promptHistory', JSON.stringify(history));
    localStorage.setItem('promptFavorites', JSON.stringify(favs));
  };

  /* ---------- helpers ---------- */
  const scrollToBottom = () =>
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(scrollToBottom, [messages]);

  /* ---------- history management ---------- */
  const addToHistory = (original, enhanced, category = "general") => {
    const newPrompt = {
      id: Date.now(),
      originalPrompt: original,
      enhancedPrompt: enhanced,
      category,
      timestamp: new Date().toISOString(),
      isFavorite: false
    };
    
    const newHistory = [newPrompt, ...promptHistory].slice(0, 100); // Keep last 100
    setPromptHistory(newHistory);
    saveToStorage(newHistory, favorites);
  };

  const toggleFavorite = (promptId) => {
    const newHistory = promptHistory.map(p => 
      p.id === promptId ? { ...p, isFavorite: !p.isFavorite } : p
    );
    const newFavorites = newHistory.filter(p => p.isFavorite);
    
    setPromptHistory(newHistory);
    setFavorites(newFavorites);
    saveToStorage(newHistory, newFavorites);
  };

  const deleteFromHistory = (promptId) => {
    const newHistory = promptHistory.filter(p => p.id !== promptId);
    const newFavorites = newHistory.filter(p => p.isFavorite);
    
    setPromptHistory(newHistory);
    setFavorites(newFavorites);
    saveToStorage(newHistory, newFavorites);
  };

  const usePrompt = (prompt) => {
    setInput(prompt.enhancedPrompt);
    setShowHistoryPanel(false);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    // You could add a toast notification here
  };

  /* ---------- filtering and search ---------- */
  const getFilteredPrompts = () => {
    let filtered = promptHistory;
    
    if (historyFilter === "favorites") {
      filtered = filtered.filter(p => p.isFavorite);
    } else if (historyFilter === "recent") {
      filtered = filtered.slice(0, 10);
    }
    
    if (historySearch.trim()) {
      filtered = filtered.filter(p => 
        p.originalPrompt.toLowerCase().includes(historySearch.toLowerCase()) ||
        p.enhancedPrompt.toLowerCase().includes(historySearch.toLowerCase()) ||
        p.category.toLowerCase().includes(historySearch.toLowerCase())
      );
    }
    
    return filtered;
  };

  const sendMessage = async () => {
    if (!input.trim() || loadingReply) return;

    const userMsg = { role: "user", text: input };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoadingReply(true);

    try {
      const { data } = await axios.post("http://localhost:8000/chat", {
        history: newMessages,
      });
      const assistantMsg = { role: "assistant", text: data.reply };
      setMessages((m) => [...m, assistantMsg]);
    } catch (err) {
      const errorMsg = { role: "assistant", text: "⚠️ Error getting response." };
      setMessages((m) => [...m, errorMsg]);
      console.error(err);
    } finally {
      setLoadingReply(false);
    }
  };

  /* ---------- enhancement ---------- */
  const enhancePrompt = async () => {
    if (!input.trim()) return;
    setLoadingEnhance(true);
    try {
      const { data } = await axios.post("http://localhost:8000/enhance", {
        prompt: input,
      });
      setEnhanced(data.enhanced_prompt);
      setPromptToSave({ original: input, enhanced: data.enhanced_prompt, category: "general" });
      setShowDialog(true);
    } catch (e) {
      console.error(e);
      setEnhanced("Enhancement failed.");
      setShowDialog(true);
    } finally {
      setLoadingEnhance(false);
    }
  };

  const handleReplace = () => {
    setInput(enhanced);
    setShowDialog(false);
    setShowSaveDialog(true); // Show save dialog after replacing
  };

  const handleRetry = () => {
    setShowDialog(false);
    enhancePrompt();
  };

  const handleSavePrompt = () => {
    addToHistory(promptToSave.original, promptToSave.enhanced, promptToSave.category);
    setShowSaveDialog(false);
  };

  return (
    <div className="prompt-enhancer-wrapper">
      {/* History Panel */}
      {showHistoryPanel && (
        <div className="history-panel">
          <div className="history-header">
            <h3><History size={18} /> Prompt History</h3>
            <button 
              className="close-panel-btn"
              onClick={() => setShowHistoryPanel(false)}
            >
              <X size={18} />
            </button>
          </div>
          
          <div className="history-controls">
            <div className="search-box">
              <Search size={16} />
              <input
                type="text"
                placeholder="Search prompts..."
                value={historySearch}
                onChange={(e) => setHistorySearch(e.target.value)}
              />
            </div>
            
            <div className="filter-tabs">
              <button 
                className={historyFilter === "all" ? "active" : ""}
                onClick={() => setHistoryFilter("all")}
              >
                All
              </button>
              <button 
                className={historyFilter === "favorites" ? "active" : ""}
                onClick={() => setHistoryFilter("favorites")}
              >
                <Star size={14} /> Favorites
              </button>
              <button 
                className={historyFilter === "recent" ? "active" : ""}
                onClick={() => setHistoryFilter("recent")}
              >
                Recent
              </button>
            </div>
          </div>
          
          <div className="history-list">
            {getFilteredPrompts().length === 0 ? (
              <div className="empty-history">
                <BookOpen size={48} />
                <p>No prompts found</p>
                <span>Start enhancing prompts to build your history!</span>
              </div>
            ) : (
              getFilteredPrompts().map((prompt) => (
                <div key={prompt.id} className="history-item">
                  <div className="prompt-preview">
                    <div className="original-prompt">
                      <span className="label">Original:</span>
                      <p>{prompt.originalPrompt.substring(0, 100)}...</p>
                    </div>
                    <div className="enhanced-prompt">
                      <span className="label">Enhanced:</span>
                      <p>{prompt.enhancedPrompt.substring(0, 100)}...</p>
                    </div>
                  </div>
                  
                  <div className="prompt-meta">
                    <span className="category-tag">
                      <Tag size={12} /> {prompt.category}
                    </span>
                    <span className="timestamp">
                      {new Date(prompt.timestamp).toLocaleDateString()}
                    </span>
                  </div>
                  
                  <div className="prompt-actions">
                    <button
                      className="action-btn"
                      onClick={() => usePrompt(prompt)}
                      title="Use this prompt"
                    >
                      Use
                    </button>
                    <button
                      className="action-btn"
                      onClick={() => copyToClipboard(prompt.enhancedPrompt)}
                      title="Copy enhanced prompt"
                    >
                      <Copy size={14} />
                    </button>
                    <button
                      className={`action-btn ${prompt.isFavorite ? 'favorited' : ''}`}
                      onClick={() => toggleFavorite(prompt.id)}
                      title="Toggle favorite"
                    >
                      <Star size={14} />
                    </button>
                    <button
                      className="action-btn delete-btn"
                      onClick={() => deleteFromHistory(prompt.id)}
                      title="Delete"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      <div className={`chat-wrapper ${showHistoryPanel ? 'with-history-panel' : ''}`}>
        <header className="chat-header">
          <button 
            className="back-btn" 
            onClick={onBackToTools}
            title="Back to AI Tools"
          >
            <ArrowLeft size={20} />
          </button>
          <span>Prompt Enhancer Chat</span>
          <div className="header-actions">
            <button
              className={`history-toggle-btn ${showHistoryPanel ? 'active' : ''}`}
              onClick={() => setShowHistoryPanel(!showHistoryPanel)}
              title="Toggle history panel"
            >
              <History size={18} />
              {promptHistory.length > 0 && (
                <span className="history-count">{promptHistory.length}</span>
              )}
            </button>
          </div>
        </header>

      <div className="chat-body">
        {messages.map((msg, i) => (
          <div key={i} className={`msg ${msg.role}`}>
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                code: ({node, inline, className, children, ...props}) => {
                  return (
                    <code className={`${className || ''} ${inline ? 'inline-code' : 'code-block'}`} {...props}>
                      {children}
                    </code>
                  )
                }
              }}
            >
              {msg.text}
            </ReactMarkdown>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="chat-input-area">
        <div className="input-wrapper">
          <textarea
            rows="2"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !loadingReply) {
                e.preventDefault();
                sendMessage();
              }
            }}
            placeholder="Type your prompt…"
            disabled={loadingReply}
          />
          <button
            className="enhance-btn"
            onClick={enhancePrompt}
            disabled={loadingEnhance || loadingReply}
            title="Enhance prompt"
          >
            {loadingEnhance ? <RotateCw size={18} className="spinning" /> : <Wand2 size={18} />}
          </button>
        </div>
        <button className="send-btn" onClick={sendMessage} disabled={loadingReply}>
          {loadingReply ? "Thinking..." : (
            <>
              Send <Send size={16} style={{ marginLeft: 4 }} />
            </>
          )}
        </button>
      </div>

      {/* ---------- enhancement dialog ---------- */}
      {showDialog && (
        <div className="dialog-overlay">
          <div className="dialog">
            <h3>Enhanced Prompt</h3>
            <textarea readOnly value={enhanced} rows={5} />
            <div className="dialog-actions">
              <button onClick={handleRetry} title="Retry">
                <RotateCw size={18} /> Retry
              </button>
              <button onClick={() => setShowDialog(false)} title="Cancel">
                <X size={18} /> Cancel
              </button>
              <button className="primary" onClick={handleReplace} title="Replace">
                <Check size={18} /> Replace
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ---------- save prompt dialog ---------- */}
      {showSaveDialog && (
        <div className="dialog-overlay">
          <div className="dialog save-dialog">
            <h3>Save Enhanced Prompt</h3>
            <p>Would you like to save this prompt to your history?</p>
            
            <div className="save-form">
              <label>Category:</label>
              <select 
                value={promptToSave.category} 
                onChange={(e) => setPromptToSave({...promptToSave, category: e.target.value})}
              >
                <option value="general">General</option>
                <option value="coding">Coding</option>
                <option value="writing">Writing</option>
                <option value="analysis">Analysis</option>
                <option value="creative">Creative</option>
                <option value="business">Business</option>
              </select>
            </div>
            
            <div className="dialog-actions">
              <button onClick={() => setShowSaveDialog(false)} title="Skip">
                Skip
              </button>
              <button className="primary" onClick={handleSavePrompt} title="Save">
                <Star size={18} /> Save to History
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
