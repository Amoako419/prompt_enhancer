import { useState } from "react";
import axios from "axios";
import { Database, ArrowLeft, Copy, RotateCw, Wand2, File, Server } from "lucide-react";
import "../App.css";
import "../styles/SqlConverter.css";

export default function SqlConverter({ onBackToTools }) {
  const [englishQuery, setEnglishQuery] = useState("");
  const [sqlResult, setSqlResult] = useState("");
  const [tsqlResult, setTsqlResult] = useState("");
  const [mongoResult, setMongoResult] = useState("");
  const [activeTab, setActiveTab] = useState("sql"); // "sql", "tsql", "mongo"
  const [loading, setLoading] = useState({
    sql: false,
    tsql: false,
    mongo: false
  });
  const [copied, setCopied] = useState(false);

  const convertToSql = async () => {
    if (!englishQuery.trim()) return;
    
    setLoading(prev => ({ ...prev, sql: true }));
    try {
      const { data } = await axios.post("http://localhost:8000/to-sql", {
        english_query: englishQuery,
      });
      setSqlResult(data.sql_query);
      setActiveTab("sql");
    } catch (err) {
      console.error(err);
      setSqlResult("Error converting to SQL. Please try again.");
    } finally {
      setLoading(prev => ({ ...prev, sql: false }));
    }
  };

  const convertToTsql = async () => {
    if (!englishQuery.trim()) return;
    
    setLoading(prev => ({ ...prev, tsql: true }));
    try {
      // You'll need to implement this endpoint in the backend
      const { data } = await axios.post("http://localhost:8000/to-tsql", {
        english_query: englishQuery,
      });
      setTsqlResult(data?.tsql_query || "T-SQL conversion is not yet implemented on the backend.");
      setActiveTab("tsql");
    } catch (err) {
      console.error(err);
      setTsqlResult("Error converting to T-SQL. Backend endpoint may not be implemented yet.");
    } finally {
      setLoading(prev => ({ ...prev, tsql: false }));
    }
  };

  const convertToMongo = async () => {
    if (!englishQuery.trim()) return;
    
    setLoading(prev => ({ ...prev, mongo: true }));
    try {
      // You'll need to implement this endpoint in the backend
      const { data } = await axios.post("http://localhost:8000/to-mongo", {
        english_query: englishQuery,
      });
      setMongoResult(data?.mongo_query || "MongoDB conversion is not yet implemented on the backend.");
      setActiveTab("mongo");
    } catch (err) {
      console.error(err);
      setMongoResult("Error converting to MongoDB. Backend endpoint may not be implemented yet.");
    } finally {
      setLoading(prev => ({ ...prev, mongo: false }));
    }
  };

  const copyToClipboard = async () => {
    let contentToCopy;
    
    switch(activeTab) {
      case "sql":
        contentToCopy = sqlResult;
        break;
      case "tsql":
        contentToCopy = tsqlResult;
        break;
      case "mongo":
        contentToCopy = mongoResult;
        break;
      default:
        contentToCopy = sqlResult;
    }
    
    if (!contentToCopy) return;
    
    try {
      await navigator.clipboard.writeText(contentToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy: ', err);
    }
  };

  const clearAll = () => {
    setEnglishQuery("");
    setSqlResult("");
    setTsqlResult("");
    setMongoResult("");
    setCopied(false);
    setActiveTab("sql");
  };

  return (
    <div className="sql-converter-wrapper">
      <header className="chat-header">
        <div className="header-left">
          <button 
            className="back-btn" 
            onClick={onBackToTools}
            title="Back to AI Tools"
          >
            <ArrowLeft size={20} />
            <span className="back-text">Back to Tools</span>
          </button>
          <span>English to SQL Converter</span>
        </div>
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
                disabled={loading.sql || !englishQuery.trim()}
              >
                {loading.sql ? (
                  <>
                    <RotateCw size={16} className="spinning" />
                    Converting to SQL...
                  </>
                ) : (
                  <>
                    <Wand2 size={16} />
                    Convert to SQL
                  </>
                )}
              </button>
              <button 
                className="convert-btn secondary"
                onClick={convertToTsql}
                disabled={loading.tsql || !englishQuery.trim()}
              >
                {loading.tsql ? (
                  <>
                    <RotateCw size={16} className="spinning" />
                    Converting to T-SQL...
                  </>
                ) : (
                  <>
                    <File size={16} />
                    Convert to T-SQL
                  </>
                )}
              </button>
              <button 
                className="convert-btn tertiary"
                onClick={convertToMongo}
                disabled={loading.mongo || !englishQuery.trim()}
              >
                {loading.mongo ? (
                  <>
                    <RotateCw size={16} className="spinning" />
                    Converting to MongoDB...
                  </>
                ) : (
                  <>
                    <Server size={16} />
                    Convert to MongoDB
                  </>
                )}
              </button>
              <button 
                className="clear-btn"
                onClick={clearAll}
                disabled={loading.sql || loading.tsql || loading.mongo}
              >
                Clear All
              </button>
            </div>
          </div>

          {/* Output Section */}
          {(sqlResult || tsqlResult || mongoResult) && (
            <div className="output-section">
              <div className="query-tabs">
                {sqlResult && (
                  <button 
                    className={`tab-btn ${activeTab === 'sql' ? 'active' : ''}`}
                    onClick={() => setActiveTab('sql')}
                  >
                    <Database size={16} />
                    SQL Query
                  </button>
                )}
                {tsqlResult && (
                  <button 
                    className={`tab-btn ${activeTab === 'tsql' ? 'active' : ''}`}
                    onClick={() => setActiveTab('tsql')}
                  >
                    <File size={16} />
                    T-SQL Query
                  </button>
                )}
                {mongoResult && (
                  <button 
                    className={`tab-btn ${activeTab === 'mongo' ? 'active' : ''}`}
                    onClick={() => setActiveTab('mongo')}
                  >
                    <Server size={16} />
                    MongoDB Query
                  </button>
                )}
              </div>
              
              <div className="section-header">
                <Database className="section-icon" size={20} />
                <h3>
                  {activeTab === 'sql' ? 'Generated SQL Query' : 
                   activeTab === 'tsql' ? 'Generated T-SQL Query' : 
                   'Generated MongoDB Query'}
                </h3>
                <button 
                  className="copy-btn"
                  onClick={copyToClipboard}
                  title="Copy to clipboard"
                >
                  <Copy size={16} />
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
              
              <div className={`query-output ${activeTab}-output`} style={{ backgroundColor: "#f8f9fa" }}>
                <pre style={{ color: "#000000" }}><code style={{ color: "#000000" }}>
                  {activeTab === 'sql' ? sqlResult : 
                   activeTab === 'tsql' ? tsqlResult : 
                   mongoResult}
                </code></pre>
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
