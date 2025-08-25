import React, { useState } from "react";
import axios from "axios";
import { ArrowLeft, Wand2, Send, History, Check, X, RotateCw, Plus } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import "../styles/PromptEnhancerModern.css";

export default function PromptEnhancerSimple({ onBackToTools }) {
  const [input, setInput] = useState("");
  const [showHistoryPanel, setShowHistoryPanel] = useState(true);
  const [messages, setMessages] = useState([]);
  const [loadingEnhance, setLoadingEnhance] = useState(false);
  const [loadingReply, setLoadingReply] = useState(false);
  const [enhanced, setEnhanced] = useState("");
  const [showDialog, setShowDialog] = useState(false);

  // Direct submit without enhancement
  const handleDirectSubmit = async () => {
    if (!input.trim()) return;
    
    const userMessage = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoadingReply(true);

    try {
      // Convert messages to the format expected by backend
      const history = [...messages, userMessage].map(msg => ({
        role: msg.role,
        text: msg.content
      }));

      const response = await axios.post("http://localhost:8000/chat", {
        history: history,
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

  // Enhance prompt with AI
  const handleEnhance = async () => {
    if (!input.trim()) return;
    
    setLoadingEnhance(true);
    try {
      const response = await axios.post("http://localhost:8000/enhance", {
        prompt: input,
      });
      
      setEnhanced(response.data.enhanced_prompt);
      setShowDialog(true);
    } catch (error) {
      console.error("Error enhancing prompt:", error);
      alert("Failed to enhance prompt. Please try again.");
    } finally {
      setLoadingEnhance(false);
    }
  };

  // Accept enhanced prompt and send to AI
  const handleAcceptEnhancement = async () => {
    const userMessage = { role: "user", content: enhanced }; // Use enhanced prompt
    setMessages(prev => [...prev, userMessage]);
    setShowDialog(false);
    setInput("");
    setLoadingReply(true);

    try {
      // Send the enhanced prompt to AI
      const history = [...messages, userMessage].map(msg => ({
        role: msg.role,
        text: msg.content
      }));

      const response = await axios.post("http://localhost:8000/chat", {
        history: history,
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

  // Reject enhanced prompt and go back
  const handleRejectEnhancement = () => {
    setShowDialog(false);
    setEnhanced("");
    // Keep the original input so user can try again
  };

  return (
    <>
      <div className="prompt-enhancer-modern">
        {/* Header */}
        <div className="modern-header">
          <div className="header-left">
            <button onClick={onBackToTools} className="back-btn-modern">
              <ArrowLeft size={18} />
              Back to Tools
            </button>
            
            <div className="logo-section">
              <div className="logo-icon">
                <Wand2 size={18} />
              </div>
              <span className="logo-text">Prompt Enhancer</span>
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
          </div>
        </div>

        {/* Body */}
        <div className="enhancer-body">
          {/* History Sidebar */}
          {showHistoryPanel && (
            <div className="history-sidebar">
              <div className="sidebar-header">
                <div className="sidebar-title">
                  <History size={16} />
                  History
                </div>
              </div>
              <div className="sidebar-content">
                <div style={{ padding: '1rem', textAlign: 'center', color: '#6b7280' }}>
                  No prompts yet
                </div>
              </div>
            </div>
          )}

          {/* Main Content */}
          <div className="main-content">
            {messages.length === 0 ? (
              /* Welcome Section */
              <div className="welcome-section">
                <h1 className="welcome-title">What's on your mind today?</h1>
                <p className="welcome-subtitle">
                  Transform your ideas into powerful, effective prompts with AI assistance.
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
              </div>
            )}

            {/* Input Section */}
            <div className="peinput-section">
              <div className="input-container">
                <div className="input-header">
                  <Plus size={16} />
                  <span></span>
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
                  placeholder=""
                  rows={1}
                />
                
                <div className="input-footer">
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
                    
                    <button
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
                    </button>
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
                <button 
                  onClick={handleEnhance} 
                  disabled={loadingEnhance}
                  className="dialog-btn secondary"
                >
                  <RotateCw size={16} />
                  {loadingEnhance ? 'Retrying...' : 'Try Again'}
                </button>
                <button onClick={handleAcceptEnhancement} className="dialog-btn primary">
                  <Check size={16} />
                  Accept & Continue
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
