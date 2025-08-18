import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { Wand2, RotateCw, Check, X, Send, ArrowLeft } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import "../App.css";

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

  /* ---------- helpers ---------- */
  const scrollToBottom = () =>
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(scrollToBottom, [messages]);

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
  };

  const handleRetry = () => {
    setShowDialog(false);
    enhancePrompt();
  };

  return (
    <div className="chat-wrapper">
      <header className="chat-header">
        <button 
          className="back-btn" 
          onClick={onBackToTools}
          title="Back to AI Tools"
        >
          <ArrowLeft size={20} />
        </button>
        <span>Prompt Enhancer Chat</span>
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
    </div>
  );
}
