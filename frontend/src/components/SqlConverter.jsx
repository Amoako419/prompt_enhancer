import { useState } from "react";
import axios from "axios";
import { Database, ArrowLeft, Copy, RotateCw, Wand2 } from "lucide-react";
import "../App.css";
import "../styles/SqlConverter.css";

export default function SqlConverter({ onBackToTools }) {
  const [englishQuery, setEnglishQuery] = useState("");
  const [sqlResult, setSqlResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const convertToSql = async () => {
    if (!englishQuery.trim()) return;
    
    setLoading(true);
    try {
      const { data } = await axios.post("http://localhost:8000/to-sql", {
        english_query: englishQuery,
      });
      setSqlResult(data.sql_query);
    } catch (err) {
      console.error(err);
      setSqlResult("Error converting to SQL. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = async () => {
    if (!sqlResult) return;
    
    try {
      await navigator.clipboard.writeText(sqlResult);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy: ', err);
    }
  };

  const clearAll = () => {
    setEnglishQuery("");
    setSqlResult("");
    setCopied(false);
  };

  return (
    <div className="sql-converter-wrapper">
      <header className="chat-header">
        <button 
          className="back-btn" 
          onClick={onBackToTools}
          title="Back to AI Tools"
        >
          <ArrowLeft size={20} />
        </button>
        <span>English to SQL Converter</span>
      </header>

      <div className="converter-container">
        <div className="converter-content">
          {/* Input Section */}
          <div className="input-section">
            <div className="section-header">
              <Database className="section-icon" size={20} />
              <h3>Describe your query in plain English</h3>
            </div>
            <textarea
              className="english-input"
              value={englishQuery}
              onChange={(e) => setEnglishQuery(e.target.value)}
              placeholder="Example: Show me all customers who placed orders in the last 30 days with their total order amounts"
              rows={4}
            />
            <div className="input-actions">
              <button 
                className="convert-btn primary"
                onClick={convertToSql}
                disabled={loading || !englishQuery.trim()}
              >
                {loading ? (
                  <>
                    <RotateCw size={16} className="spinning" />
                    Converting...
                  </>
                ) : (
                  <>
                    <Wand2 size={16} />
                    Convert to SQL
                  </>
                )}
              </button>
              <button 
                className="clear-btn"
                onClick={clearAll}
                disabled={loading}
              >
                Clear All
              </button>
            </div>
          </div>

          {/* Output Section */}
          {sqlResult && (
            <div className="output-section">
              <div className="section-header">
                <Database className="section-icon" size={20} />
                <h3>Generated SQL Query</h3>
                <button 
                  className="copy-btn"
                  onClick={copyToClipboard}
                  title="Copy to clipboard"
                >
                  <Copy size={16} />
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
              <div className="sql-output">
                <pre><code>{sqlResult}</code></pre>
              </div>
            </div>
          )}

          {/* Examples Section */}
          <div className="examples-section">
            <h3>Example Queries</h3>
            <div className="examples-grid">
              <div 
                className="example-card"
                onClick={() => setEnglishQuery("Find all users who registered in the last week")}
              >
                <strong>User Registration:</strong>
                <span>Find all users who registered in the last week</span>
              </div>
              <div 
                className="example-card"
                onClick={() => setEnglishQuery("Show the top 10 products by sales revenue")}
              >
                <strong>Sales Analysis:</strong>
                <span>Show the top 10 products by sales revenue</span>
              </div>
              <div 
                className="example-card"
                onClick={() => setEnglishQuery("Get customers with more than 5 orders")}
              >
                <strong>Customer Analysis:</strong>
                <span>Get customers with more than 5 orders</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
