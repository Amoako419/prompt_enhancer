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

export default function PromptEnhancerModern({ onBackToTools }) {
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
  const [showHistoryPanel, setShowHistoryPanel] = useState(true); // Open by default
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

  /* ---------- text formatting functions ---------- */
  const cleanText = (text) => {
    return text
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/`(.*?)`/g, '$1')
      .replace(/#{1,6}\s/g, '')
      .replace(/^\s*[-*+]\s/gm, '')
      .replace(/^\s*\d+\.\s/gm, '')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/[.]{2,}/g, '.')
      .replace(/[!]{2,}/g, '!')
      .replace(/[?]{2,}/g, '?')
      .trim();
  };

  /* ---------- scroll to bottom ---------- */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loadingReply]);

  /* ---------- filter history ---------- */
  const getFilteredHistory = () => {
    let filtered = [...promptHistory];
    
    if (historyFilter === "favorites") {
      filtered = filtered.filter(item => favorites.includes(item.id));
    } else if (historyFilter === "recent") {
      filtered = filtered.slice(-10);
    }
    
    if (historySearch) {
      filtered = filtered.filter(item => 
        item.original.toLowerCase().includes(historySearch.toLowerCase()) ||
        item.enhanced.toLowerCase().includes(historySearch.toLowerCase())
      );
    }
    
    return filtered.reverse();
  };

  /* ---------- chat submission ---------- */
  const handleSubmit = async () => {
    if (!input.trim() || loadingReply) return;
    
    const userMessage = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoadingReply(true);

    try {
      const response = await axios.post("http://localhost:8000/chat", {
        messages: [...messages, userMessage],
      });
      
      const assistantMessage = { role: "assistant", content: response.data.reply };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage = { 
        role: "assistant", 
        content: "Sorry, I encountered an error. Please try again." 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoadingReply(false);
    }
  };

  /* ---------- direct submit without enhancement ---------- */
  const handleDirectSubmit = async () => {
    if (!input.trim()) return;
    
    const userMessage = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoadingReply(true);

    try {
      const response = await axios.post("http://localhost:8000/chat", {
        messages: [...messages, userMessage],
      });
      
      const assistantMessage = { role: "assistant", content: response.data.reply };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage = { 
        role: "assistant", 
        content: "Sorry, I encountered an error. Please try again." 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoadingReply(false);
    }
  };

  /* ---------- prompt enhancement ---------- */
  const handleEnhance = async () => {
    if (!input.trim()) return;
    
    setLoadingEnhance(true);
    try {
      const response = await axios.post("http://localhost:8000/enhance", {
        prompt: input,
      });
      
      setEnhanced(response.data.enhanced);
      setShowDialog(true);
    } catch (error) {
      console.error("Error enhancing prompt:", error);
      alert("Failed to enhance prompt. Please try again.");
    } finally {
      setLoadingEnhance(false);
    }
  };

  /* ---------- history management ---------- */
  const addToHistory = (original, enhanced, category = "general") => {
    const newItem = {
      id: Date.now(),
      original: cleanText(original),
      enhanced: cleanText(enhanced),
      category,
      timestamp: Date.now()
    };
    
    const updatedHistory = [newItem, ...promptHistory];
    setPromptHistory(updatedHistory);
    saveToStorage(updatedHistory, favorites);
  };

  const toggleFavorite = (id) => {
    const updatedFavorites = favorites.includes(id)
      ? favorites.filter(fav => fav !== id)
      : [...favorites, id];
    
    setFavorites(updatedFavorites);
    saveToStorage(promptHistory, updatedFavorites);
  };

  const deleteFromHistory = (id) => {
    const updatedHistory = promptHistory.filter(item => item.id !== id);
    const updatedFavorites = favorites.filter(fav => fav !== id);
    
    setPromptHistory(updatedHistory);
    setFavorites(updatedFavorites);
    saveToStorage(updatedHistory, updatedFavorites);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    // Could add a toast notification here
  };

  /* ---------- dialog actions ---------- */
  const handleAcceptEnhancement = () => {
    const userMessage = { role: "user", content: input };
    const assistantMessage = { role: "assistant", content: enhanced };
    
    setMessages([userMessage, assistantMessage]);
    setPromptToSave({ original: input, enhanced: enhanced, category: "general" });
    setShowDialog(false);
    setShowSaveDialog(true);
    setInput("");
  };

  const handleRejectEnhancement = () => {
    setShowDialog(false);
    setEnhanced("");
  };

  const handleSavePrompt = () => {
    addToHistory(promptToSave.original, promptToSave.enhanced, promptToSave.category);
    setShowSaveDialog(false);
  };

  return (
    <div className="prompt-enhancer-modern">
      {/* Modern Header */}
      <div className="modern-header">
        <div className="header-left">
          <button
            onClick={onBackToTools}
            className="back-btn-modern"
          >
            <ArrowLeft size={18} />
            Back to Tools
          </button>
          
          <div className="logo-section">
            <div className="logo-icon">
              <Wand2 size={18} />
            </div>
            <span className="logo-text">AI Text Processor</span>
          </div>
        </div>
        
        <div className="header-actions">
          <button 
            className="header-btn"
            onClick={() => setShowHistoryPanel(!showHistoryPanel)}
            title={showHistoryPanel ? "Hide History" : "Show History"}
          >
            <History size={18} />
          </button>
          <button 
            className="header-btn"
            title="Settings"
          >
            <Settings size={18} />
          </button>
        </div>
      </div>

      {/* Body with Sidebar */}
      <div className="enhancer-body">
        {/* History Sidebar */}
        <div className={`history-sidebar ${!showHistoryPanel ? 'collapsed' : ''}`}>
          <div className="sidebar-header">
            <div className="sidebar-title">
              <History size={16} />
              History
            </div>
          </div>
          
          <div className="sidebar-content">
            <div className="sidebar-controls">
              <input
                type="text"
                placeholder="Search prompts..."
                value={historySearch}
                onChange={(e) => setHistorySearch(e.target.value)}
                className="sidebar-search"
              />
              
              <div className="sidebar-filters">
                <button 
                  className={`filter-btn ${historyFilter === "all" ? "active" : ""}`}
                  onClick={() => setHistoryFilter("all")}
                >
                  All
                </button>
                <button 
                  className={`filter-btn ${historyFilter === "favorites" ? "active" : ""}`}
                  onClick={() => setHistoryFilter("favorites")}
                >
                  ★
                </button>
                <button 
                  className={`filter-btn ${historyFilter === "recent" ? "active" : ""}`}
                  onClick={() => setHistoryFilter("recent")}
                >
                  Recent
                </button>
              </div>
            </div>

            <div className="history-list">
              {getFilteredHistory().map((item) => (
                <div key={item.id} className="history-item" onClick={() => {
                  setInput(item.original);
                }}>
                  <div className="history-item-header">
                    <span className="history-category">{item.category}</span>
                    <div className="history-actions">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleFavorite(item.id);
                        }}
                        className={`history-action-btn favorite ${favorites.includes(item.id) ? 'active' : ''}`}
                      >
                        <Star size={12} />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          copyToClipboard(item.enhanced);
                        }}
                        className="history-action-btn"
                      >
                        <Copy size={12} />
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteFromHistory(item.id);
                        }}
                        className="history-action-btn"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                  <div className="history-text">
                    <div><strong>Original:</strong> {item.original.substring(0, 80)}...</div>
                    <div><strong>Enhanced:</strong> {item.enhanced.substring(0, 80)}...</div>
                  </div>
                  <div className="history-timestamp">
                    {new Date(item.timestamp).toLocaleDateString()}
                  </div>
                </div>
              ))}
              {getFilteredHistory().length === 0 && (
                <div style={{ textAlign: 'center', color: '#6b7280', padding: '2rem 0' }}>
                  No prompts found
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-content">
        {messages.length === 0 ? (
          /* Welcome Section */
          <div className="welcome-section">
            <h1 className="welcome-title">What's on your mind today?</h1>
            <p className="welcome-subtitle">
              Transform your ideas into powerful, effective prompts with AI assistance. 
              Get better results from any AI model with enhanced prompts.
            </p>
          </div>
        ) : (
          /* Messages Section */
          <div className="messages-container">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                <div className="message-header">
                  <span>{msg.role === 'user' ? 'You' : 'Assistant'}</span>
                  <span>{new Date().toLocaleTimeString()}</span>
                </div>
                <div className="message-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>
                </div>
              </div>
            ))}
            
            {loadingReply && (
              <div className="message assistant">
                <div className="message-header">
                  <span>Assistant</span>
                </div>
                <div className="typing-indicator">
                  <span>Thinking</span>
                  <div className="typing-dots">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Input Section */}
        <div className="peinput-section">
          <div className="input-container">
            <div className="input-header">
              <Plus size={16} />
              <span>Ask anything</span>
            </div>
            
            <textarea
              className="main-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleDirectSubmit();
                }
              }}
              placeholder="Type your message..."
              rows={2}
            />
            
            <div className="input-footer">
              <div className="input-actions">
                <button className="input-btn" title="Voice input">
                  <Mic size={16} />
                </button>
                <button className="input-btn" title="Upload file">
                  <Plus size={16} />
                </button>
              </div>
              
              <div className="action-buttons">
                <button
                  onClick={handleEnhance}
                  disabled={!input.trim() || loadingEnhance || loadingReply}
                  className="enhance-btn-icon"
                  title="Enhance with AI"
                >
                  {loadingEnhance ? (
                    <span className="loading-spinner"></span>
                  ) : (
                    <Wand2 size={16} />
                  )}
                </button>
                
                {/* <button
                  onClick={handleDirectSubmit}
                  disabled={!input.trim() || loadingEnhance || loadingReply}
                  className="send-btn"
                  title="Send message"
                >
                  {loadingReply ? (
                    <span className="loading-spinner"></span>
                  ) : (
                    <Send size={16} />
                  )}
                </button> */}
              </div>
            </div>
          </div>
        </div>
        </div>
      </div>

      {/* Enhancement Dialog */}
      {showDialog && (
        <div className="enhancement-dialog">
          <div className="dialog-content">
            <div className="dialog-header">
              <h2 className="dialog-title">Enhanced Prompt</h2>
              <button onClick={handleRejectEnhancement} className="header-btn">
                <X size={18} />
              </button>
            </div>
            
            <div className="dialog-body">
              <div className="comparison-section">
                <h3 className="comparison-title">Original Prompt</h3>
                <div className="text-box">{input}</div>
              </div>
              
              <div className="comparison-section">
                <h3 className="comparison-title">Enhanced Prompt</h3>
                <div className="text-box">{enhanced}</div>
              </div>
            </div>
            
            <div className="dialog-actions">
              <button onClick={handleRejectEnhancement} className="dialog-btn secondary">
                <X size={16} />
                Reject
              </button>
              <button onClick={handleAcceptEnhancement} className="dialog-btn primary">
                <Check size={16} />
                Accept & Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Save Dialog */}
      {showSaveDialog && (
        <div className="enhancement-dialog">
          <div className="dialog-content">
            <div className="dialog-header">
              <h2 className="dialog-title">Save to History</h2>
              <button onClick={() => setShowSaveDialog(false)} className="header-btn">
                <X size={18} />
              </button>
            </div>
            
            <div className="dialog-body">
              <div className="comparison-section">
                <h3 className="comparison-title">Category</h3>
                <select
                  value={promptToSave.category}
                  onChange={(e) => setPromptToSave(prev => ({ ...prev, category: e.target.value }))}
                  className="main-input"
                  style={{ minHeight: 'auto', padding: '0.5rem' }}
                >
                  <option value="general">General</option>
                  <option value="creative">Creative Writing</option>
                  <option value="technical">Technical</option>
                  <option value="business">Business</option>
                  <option value="educational">Educational</option>
                </select>
              </div>
            </div>
            
            <div className="dialog-actions">
              <button onClick={() => setShowSaveDialog(false)} className="dialog-btn secondary">
                Cancel
              </button>
              <button onClick={handleSavePrompt} className="dialog-btn primary">
                <BookOpen size={16} />
                Save to History
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
