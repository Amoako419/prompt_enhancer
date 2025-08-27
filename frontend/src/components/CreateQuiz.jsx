import { useState } from "react";
import axios from "axios";
import { 
  ArrowLeft, Plus, Trash2, Save, X, CheckSquare, 
  FileText, AlertCircle, Edit, Check, Brain,
  BarChart3, Code, Database, TrendingUp, Sparkles, Loader,
  AlertTriangle
} from "lucide-react";
import "../styles/CreateQuiz.css";

export default function CreateQuiz({ onBack, onQuizCreated }) {
  const [createMethod, setCreateMethod] = useState("manual"); // "manual" or "ai"
  const [quizData, setQuizData] = useState({
    title: "",
    description: "",
    difficulty: "beginner",
    duration: 15,
    questions: []
  });
  
  const [currentQuestion, setCurrentQuestion] = useState({
    question: "",
    options: ["", "", "", ""],
    correct_answer: 0,
    explanation: "",
    type: "multiple_choice"
  });
  
  const [aiSettings, setAiSettings] = useState({
    category: "python",
    difficulty: "medium",
    numQuestions: 5,
    topic: ""
  });
  
  const [editingQuestionIndex, setEditingQuestionIndex] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [aiGenerating, setAiGenerating] = useState(false);
  const [error, setError] = useState("");
  const [aiError, setAiError] = useState("");
  const [saveSuccess, setSaveSuccess] = useState(false);
  
  const difficultyOptions = [
    { value: "beginner", label: "Beginner", color: "#2ecc71" },
    { value: "intermediate", label: "Intermediate", color: "#f39c12" },
    { value: "advanced", label: "Advanced", color: "#e74c3c" }
  ];
  
  const categories = [
    { id: 'python', title: 'Python for Data Science', icon: <Code size={18} />, color: '#3776ab' },
    { id: 'sql', title: 'SQL & Database Design', icon: <Database size={18} />, color: '#336791' },
    { id: 'statistics', title: 'Statistics & Probability', icon: <BarChart3 size={18} />, color: '#e74c3c' },
    { id: 'machine_learning', title: 'Machine Learning', icon: <Brain size={18} />, color: '#9b59b6' },
    { id: 'data_visualization', title: 'Data Visualization', icon: <TrendingUp size={18} />, color: '#2ecc71' },
    { id: 'data_engineering', title: 'Data Engineering', icon: <FileText size={18} />, color: '#f39c12' }
  ];
  
  const handleQuizChange = (field, value) => {
    setQuizData({ ...quizData, [field]: value });
  };
  
  const handleQuestionChange = (field, value) => {
    setCurrentQuestion({ ...currentQuestion, [field]: value });
  };
  
  const handleOptionChange = (index, value) => {
    const newOptions = [...currentQuestion.options];
    newOptions[index] = value;
    setCurrentQuestion({ ...currentQuestion, options: newOptions });
  };
  
  const addQuestion = () => {
    // Validate question
    if (!currentQuestion.question.trim()) {
      setError("Question text cannot be empty");
      return;
    }
    
    if (currentQuestion.options.some(opt => !opt.trim())) {
      setError("All options must have content");
      return;
    }
    
    setError("");
    
    if (editingQuestionIndex >= 0) {
      // Update existing question
      const updatedQuestions = [...quizData.questions];
      updatedQuestions[editingQuestionIndex] = { ...currentQuestion };
      setQuizData({ ...quizData, questions: updatedQuestions });
      setEditingQuestionIndex(-1);
    } else {
      // Add new question
      setQuizData({ 
        ...quizData, 
        questions: [...quizData.questions, { ...currentQuestion }] 
      });
    }
    
    // Reset current question
    setCurrentQuestion({
      question: "",
      options: ["", "", "", ""],
      correct_answer: 0,
      explanation: "",
      type: "multiple_choice"
    });
  };
  
  const editQuestion = (index) => {
    setCurrentQuestion({ ...quizData.questions[index] });
    setEditingQuestionIndex(index);
  };
  
  const deleteQuestion = (index) => {
    const updatedQuestions = [...quizData.questions];
    updatedQuestions.splice(index, 1);
    setQuizData({ ...quizData, questions: updatedQuestions });
    
    // If deleting the question being edited, reset the form
    if (index === editingQuestionIndex) {
      setCurrentQuestion({
        question: "",
        options: ["", "", "", ""],
        correct_answer: 0,
        explanation: "",
        type: "multiple_choice"
      });
      setEditingQuestionIndex(-1);
    }
  };
  
  const handleAiSettingChange = (field, value) => {
    setAiQuizSettings({ ...aiQuizSettings, [field]: value });
  };
  
  const generateAiQuiz = async () => {
    // Validate settings
    if (!quizData.title.trim()) {
      setAiError("Quiz title is required");
      return;
    }
    
    if (!aiSettings.category) {
      setAiError("Please select a category");
      return;
    }
    
    setAiError("");
    setAiGenerating(true);
    
    try {
      // Call the backend to generate quiz questions
      const { data } = await axios.post("http://localhost:8000/custom-quiz/generate", {
        category: aiSettings.category,
        difficulty: aiSettings.difficulty,
        numQuestions: aiSettings.numQuestions,
        topic: aiSettings.topic,
        title: quizData.title
      });
      
      // Format questions to ensure consistency between "question" and "text" fields
      const formattedQuestions = data.questions.map(q => ({
        text: q.question,
        question: q.question,
        options: q.options,
        correctIndex: q.correct_index,
        correct_answer: q.correct_index, // For compatibility with manual format
        explanation: q.explanation || ""
      }));
      
      // Update quiz data with generated content
      setQuizData({
        ...quizData,
        description: `AI-generated quiz on ${getCategoryTitle(aiSettings.category)}`,
        difficulty: aiSettings.difficulty,
        duration: aiSettings.numQuestions * 2, // 2 minutes per question as a default
        questions: formattedQuestions
      });
    } catch (err) {
      console.error(err);
      setAiError("Failed to generate quiz. Please try again.");
    } finally {
      setAiGenerating(false);
    }
  };
  
  const getCategoryTitle = (categoryId) => {
    const category = categories.find(cat => cat.id === categoryId);
    return category ? category.title : categoryId;
  };
  
  const saveQuiz = async () => {
    // Validate quiz data
    if (!quizData.title.trim()) {
      setError("Quiz title is required");
      return;
    }
    
    if (quizData.questions.length === 0) {
      setError("Quiz must have at least one question");
      return;
    }
    
    setError("");
    setLoading(true);
    
    try {
      // Format questions to ensure consistency with backend format
      const formattedQuizData = {
        ...quizData,
        questions: quizData.questions.map(q => ({
          question: q.question || q.text,
          options: q.options,
          correct_answer: q.correctIndex || q.correct_answer,
          explanation: q.explanation || "",
          type: q.type || "multiple_choice"
        }))
      };
      
      await axios.post("http://localhost:8000/custom-quiz/create", formattedQuizData);
      setSaveSuccess(true);
      setTimeout(() => {
        if (onQuizCreated) {
          onQuizCreated(quizData);
        }
      }, 1500);
    } catch (err) {
      console.error(err);
      setError("Failed to save quiz. Please try again.");
    } finally {
      setLoading(false);
    }
  };
  
  const cancelEditingQuestion = () => {
    setCurrentQuestion({
      question: "",
      options: ["", "", "", ""],
      correct_answer: 0,
      explanation: "",
      type: "multiple_choice"
    });
    setEditingQuestionIndex(-1);
  };
  
  return (
    <div className="create-quiz-wrapper">
      <header className="create-quiz-header">
        <button 
          className="back-btn" 
          onClick={onBack}
          title="Back to Categories"
        >
          <ArrowLeft size={20} />
        </button>
        <span>Create Custom Quiz</span>
        <button 
          className="save-quiz-btn" 
          onClick={saveQuiz}
          disabled={loading || aiGenerating || (createMethod === "ai" && quizData.questions.length === 0)}
        >
          {loading ? <span className="spinning"><Save size={20} /></span> : <Save size={20} />}
          Save Quiz
        </button>
      </header>
      
      <div className="create-quiz-container">
        {saveSuccess ? (
          <div className="success-message">
            <CheckSquare size={48} />
            <h2>Quiz Created Successfully!</h2>
            <p>Your quiz has been saved and is now available to take.</p>
          </div>
        ) : (
          <>
            <div className="quiz-expiry-notice">
              <AlertCircle size={16} />
              <span>Note: Custom quizzes will automatically expire after completion to maintain freshness</span>
            </div>
            
            <div className="creation-method-selector">
              <button 
                className={`method-btn ${createMethod === "ai" ? "active" : ""}`}
                onClick={() => setCreateMethod("ai")}
              >
                <Sparkles size={18} />
                AI-Powered Generation
              </button>
              <button 
                className={`method-btn ${createMethod === "manual" ? "active" : ""}`}
                onClick={() => setCreateMethod("manual")}
              >
                <Edit size={18} />
                Manual Creation
              </button>
            </div>
            
            {createMethod === "ai" ? (
              <div className="ai-generation-container">
                <h3><Sparkles size={18} /> AI Quiz Generation</h3>
                <p>Generate a complete quiz using AI. Simply select your preferences below.</p>
                
                <div className="ai-settings">
                  <div className="settings-group">
                    <label htmlFor="quizTitle">Quiz Title</label>
                    <input
                      type="text"
                      id="quizTitle"
                      placeholder="Enter a title for your quiz"
                      value={quizData.title}
                      onChange={(e) => setQuizData({...quizData, title: e.target.value})}
                      required
                    />
                  </div>
                  
                  <div className="settings-row">
                    <div className="settings-group">
                      <label>Category</label>
                      <div className="category-options">
                        {categories.map((cat) => (
                          <button
                            key={cat.id}
                            className={`category-option ${aiSettings.category === cat.id ? "selected" : ""}`}
                            onClick={() => setAiSettings({...aiSettings, category: cat.id})}
                            title={cat.name}
                          >
                            {cat.icon}
                            <span>{cat.name}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                    
                    <div className="settings-group">
                      <label>Difficulty</label>
                      <div className="difficulty-options">
                        {["Easy", "Medium", "Hard"].map((diff) => (
                          <button
                            key={diff}
                            className={`difficulty-option ${aiSettings.difficulty.toLowerCase() === diff.toLowerCase() ? "selected" : ""}`}
                            onClick={() => setAiSettings({...aiSettings, difficulty: diff.toLowerCase()})}
                          >
                            {diff}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  <div className="settings-row">
                    <div className="settings-group">
                      <label htmlFor="numQuestions">Number of Questions</label>
                      <input
                        type="range"
                        id="numQuestions"
                        min="3"
                        max="15"
                        value={aiSettings.numQuestions}
                        onChange={(e) => setAiSettings({...aiSettings, numQuestions: parseInt(e.target.value)})}
                      />
                      <div className="range-value">{aiSettings.numQuestions} questions</div>
                    </div>
                    
                    <div className="settings-group">
                      <label htmlFor="customTopic">Custom Topic (Optional)</label>
                      <input
                        type="text"
                        id="customTopic"
                        placeholder="Specify a specific topic"
                        value={aiSettings.topic}
                        onChange={(e) => setAiSettings({...aiSettings, topic: e.target.value})}
                      />
                    </div>
                  </div>
                </div>
                
                <button 
                  className="generate-btn" 
                  onClick={generateAiQuiz}
                  disabled={aiGenerating || !quizData.title || !aiSettings.category}
                >
                  {aiGenerating ? (
                    <>
                      <span className="spinning"><Loader size={20} /></span>
                      Generating Quiz...
                    </>
                  ) : (
                    <>
                      <Sparkles size={20} />
                      Generate Quiz
                    </>
                  )}
                </button>
                
                {aiError && (
                  <div className="ai-error">
                    <AlertTriangle size={16} />
                    {aiError}
                  </div>
                )}
                
                {quizData.questions.length > 0 && (
                  <div className="generated-questions">
                    <h4>Generated Questions</h4>
                    <p>Your AI-generated quiz is ready! Review the questions below before saving.</p>
                    
                    {quizData.questions.map((question, qIndex) => (
                      <div className="ai-question-card" key={qIndex}>
                        <div className="question-header">
                          <span>Question {qIndex + 1}</span>
                          <button 
                            className="delete-question" 
                            onClick={() => deleteQuestion(qIndex)}
                            title="Remove Question"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                        <div className="question-text">{question.question || question.text}</div>
                        <div className="ai-options">
                          {question.options.map((option, oIndex) => (
                            <div 
                              className={`ai-option ${question.correctIndex === oIndex ? "correct" : ""}`} 
                              key={oIndex}
                            >
                              {option}
                              {question.correctIndex === oIndex && <Check size={16} />}
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <>
                {error && (
                  <div className="error-message">
                    <AlertCircle size={20} />
                    <p>{error}</p>
                  </div>
                )}
                
                <div className="quiz-details-section">
                  <h2>Quiz Details</h2>
              
              <div className="form-group">
                <label htmlFor="quiz-title">Quiz Title</label>
                <input 
                  id="quiz-title" 
                  type="text" 
                  value={quizData.title}
                  onChange={(e) => handleQuizChange("title", e.target.value)}
                  placeholder="Enter a descriptive title for your quiz"
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="quiz-description">Description</label>
                <textarea 
                  id="quiz-description" 
                  value={quizData.description}
                  onChange={(e) => handleQuizChange("description", e.target.value)}
                  placeholder="Explain what this quiz is about"
                  rows={3}
                />
              </div>
              
              <div className="form-row">
                <div className="form-group">
                  <label>Difficulty</label>
                  <div className="difficulty-options">
                    {difficultyOptions.map(option => (
                      <label 
                        key={option.value}
                        className={`difficulty-option ${quizData.difficulty === option.value ? 'selected' : ''}`}
                        style={quizData.difficulty === option.value ? { borderColor: option.color, background: `${option.color}10` } : {}}
                      >
                        <input
                          type="radio"
                          value={option.value}
                          checked={quizData.difficulty === option.value}
                          onChange={(e) => handleQuizChange("difficulty", e.target.value)}
                        />
                        <span>{option.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
                
                <div className="form-group">
                  <label htmlFor="quiz-duration">Duration (minutes)</label>
                  <input 
                    id="quiz-duration" 
                    type="number" 
                    min="1" 
                    max="120"
                    value={quizData.duration}
                    onChange={(e) => handleQuizChange("duration", parseInt(e.target.value) || 15)}
                  />
                </div>
              </div>
            </div>
            
            <div className="questions-section">
              <h2>Questions</h2>
              
              <div className="question-form">
                <div className="form-group">
                  <label htmlFor="question-text">Question Text</label>
                  <textarea 
                    id="question-text" 
                    value={currentQuestion.question}
                    onChange={(e) => handleQuestionChange("question", e.target.value)}
                    placeholder="Enter your question here"
                    rows={3}
                  />
                </div>
                
                <div className="options-container">
                  <label>Answer Options</label>
                  {currentQuestion.options.map((option, index) => (
                    <div key={index} className="option-input">
                      <div className="option-radio">
                        <input
                          type="radio"
                          name="correct-option"
                          checked={currentQuestion.correct_answer === index}
                          onChange={() => handleQuestionChange("correct_answer", index)}
                        />
                      </div>
                      <input 
                        type="text" 
                        value={option}
                        onChange={(e) => handleOptionChange(index, e.target.value)}
                        placeholder={`Option ${index + 1}`}
                      />
                    </div>
                  ))}
                  <p className="option-help">Select the radio button for the correct answer</p>
                </div>
                
                <div className="form-group">
                  <label htmlFor="explanation">Explanation (Optional)</label>
                  <textarea 
                    id="explanation" 
                    value={currentQuestion.explanation}
                    onChange={(e) => handleQuestionChange("explanation", e.target.value)}
                    placeholder="Explain why the correct answer is right (optional)"
                    rows={2}
                  />
                </div>
                
                <div className="question-actions">
                  {editingQuestionIndex >= 0 ? (
                    <>
                      <button className="btn update-question" onClick={addQuestion}>
                        <Check size={16} />
                        Update Question
                      </button>
                      <button className="btn cancel-edit" onClick={cancelEditingQuestion}>
                        <X size={16} />
                        Cancel
                      </button>
                    </>
                  ) : (
                    <button className="btn add-question" onClick={addQuestion}>
                      <Plus size={16} />
                      Add Question
                    </button>
                  )}
                </div>
              </div>
              
              {quizData.questions.length > 0 ? (
                <div className="questions-list">
                  <h3>Questions Added ({quizData.questions.length})</h3>
                  {quizData.questions.map((q, index) => (
                    <div key={index} className="question-item">
                      <div className="question-item-header">
                        <span className="question-number">Q{index + 1}</span>
                        <div className="question-actions">
                          <button 
                            className="edit-btn" 
                            onClick={() => editQuestion(index)}
                            title="Edit question"
                          >
                            <Edit size={16} />
                          </button>
                          <button 
                            className="delete-btn" 
                            onClick={() => deleteQuestion(index)}
                            title="Delete question"
                          >
                            <Trash2 size={16} />
                          </button>
                        </div>
                      </div>
                      <div className="question-item-content">
                        <p>{q.question}</p>
                        <div className="options-preview">
                          {q.options.map((opt, optIndex) => (
                            <div 
                              key={optIndex} 
                              className={`option-preview ${q.correct_answer === optIndex ? 'correct' : ''}`}
                            >
                              {q.correct_answer === optIndex && <Check size={14} />}
                              <span>{opt}</span>
                            </div>
                          ))}
                        </div>
                        {q.explanation && (
                          <div className="explanation-preview">
                            <FileText size={14} />
                            <p>{q.explanation}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-questions">
                  <p>No questions added yet. Start adding questions above.</p>
                </div>
              )}
            </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
