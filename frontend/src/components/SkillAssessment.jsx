import { useState, useEffect } from "react";
import axios from "axios";
import { 
  ArrowLeft, Brain, Clock, CheckCircle, XCircle, Award, 
  BarChart3, Code, Database, TrendingUp, FileText, Play,
  RotateCw, Trophy, Star, Target, ChevronRight, ChevronLeft,
  RotateCcw, Share2, Plus, Bookmark
} from "lucide-react";
import "../App.css";
import "../styles/SkillAssessment.css";
import CreateQuiz from "./CreateQuiz";

export default function SkillAssessment({ onBackToTools }) {
  const [currentView, setCurrentView] = useState('categories'); // 'categories', 'quiz', 'results', 'create-quiz'
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [currentQuiz, setCurrentQuiz] = useState(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [userAnswers, setUserAnswers] = useState([]);
  const [timeLeft, setTimeLeft] = useState(0);
  const [quizStarted, setQuizStarted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [customQuizzes, setCustomQuizzes] = useState([]);

  const skillCategories = [
    {
      id: 'python',
      title: 'Python for Data Science',
      icon: <Code className="category-icon" />,
      description: 'Test your Python skills including pandas, numpy, and data manipulation',
      difficulty: 'Intermediate',
      duration: '15 mins',
      questions: 10,
      color: '#3776ab'
    },
    {
      id: 'sql',
      title: 'SQL & Database Design',
      icon: <Database className="category-icon" />,
      description: 'Evaluate SQL querying, joins, optimization, and database design principles',
      difficulty: 'Intermediate',
      duration: '20 mins',
      questions: 12,
      color: '#336791'
    },
    {
      id: 'statistics',
      title: 'Statistics & Probability',
      icon: <BarChart3 className="category-icon" />,
      description: 'Test statistical concepts, hypothesis testing, and probability theory',
      difficulty: 'Advanced',
      duration: '25 mins',
      questions: 15,
      color: '#e74c3c'
    },
    {
      id: 'machine_learning',
      title: 'Machine Learning',
      icon: <Brain className="category-icon" />,
      description: 'Assess ML algorithms, model evaluation, and feature engineering skills',
      difficulty: 'Advanced',
      duration: '30 mins',
      questions: 18,
      color: '#9b59b6'
    },
    {
      id: 'data_visualization',
      title: 'Data Visualization',
      icon: <TrendingUp className="category-icon" />,
      description: 'Evaluate visualization principles, tools, and best practices',
      difficulty: 'Beginner',
      duration: '12 mins',
      questions: 8,
      color: '#2ecc71'
    },
    {
      id: 'data_engineering',
      title: 'Data Engineering',
      icon: <FileText className="category-icon" />,
      description: 'Test ETL processes, data pipelines, and big data technologies',
      difficulty: 'Advanced',
      duration: '35 mins',
      questions: 20,
      color: '#f39c12'
    }
  ];

  // Timer effect
  // Load custom quizzes on component mount
  useEffect(() => {
    loadCustomQuizzes();
  }, []);

  // Timer effect
  useEffect(() => {
    let interval = null;
    if (quizStarted && timeLeft > 0) {
      interval = setInterval(() => {
        setTimeLeft(timeLeft - 1);
      }, 1000);
    } else if (timeLeft === 0 && quizStarted) {
      handleQuizComplete();
    }
    return () => clearInterval(interval);
  }, [quizStarted, timeLeft]);

  const loadCustomQuizzes = async () => {
    try {
      // In a real application, fetch from backend
      // For now, we'll get from localStorage as a fallback
      const savedQuizzes = localStorage.getItem('customQuizzes');
      if (savedQuizzes) {
        setCustomQuizzes(JSON.parse(savedQuizzes));
      }

      // If backend endpoint is available, use it
      try {
        const { data } = await axios.get("http://localhost:8000/custom-quiz/list");
        setCustomQuizzes(data);
      } catch (err) {
        // Backend endpoint not available, continue with localStorage data
        console.log("Using localStorage for custom quizzes");
      }
    } catch (err) {
      console.error("Error loading custom quizzes:", err);
    }
  };
  
  const handleQuizCreated = (newQuiz) => {
    // Add ID to the new quiz
    const quizWithId = {
      ...newQuiz,
      id: `custom-${Date.now()}`,
      isCustom: true
    };
    
    // Update state
    const updatedQuizzes = [...customQuizzes, quizWithId];
    setCustomQuizzes(updatedQuizzes);
    
    // Save to localStorage as fallback
    localStorage.setItem('customQuizzes', JSON.stringify(updatedQuizzes));
    
    // Return to categories view
    setCurrentView('categories');
  };

  const deleteQuiz = async (quizId, e) => {
    // Stop the click event from bubbling up to the card
    e.stopPropagation();
    
    try {
      // Filter out the quiz with the matching ID
      const updatedQuizzes = customQuizzes.filter(quiz => quiz.id !== quizId);
      
      // Update state
      setCustomQuizzes(updatedQuizzes);
      
      // Update localStorage
      localStorage.setItem('customQuizzes', JSON.stringify(updatedQuizzes));
      
      // If backend endpoint is available, delete from there too
      try {
        await axios.delete(`http://localhost:8000/custom-quiz/delete/${quizId}`);
      } catch (err) {
        console.log("Using only localStorage for deleting quiz");
      }
    } catch (err) {
      console.error("Error deleting quiz:", err);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const startQuiz = async (category) => {
    setLoading(true);
    
    try {
      // Check if this is a custom quiz or a predefined category
      if (category.isCustom) {
        // For custom quizzes, we already have the questions
        const customQuizData = {
          questions: category.questions,
          category: category.title,
          difficulty: category.difficulty,
          duration_minutes: category.duration
        };
        
        setCurrentQuiz(customQuizData);
        setSelectedCategory({
          ...category,
          title: category.title,
          id: category.id,
          difficulty: category.difficulty,
          duration: `${category.duration} mins`,
          questions: category.questions.length
        });
        setCurrentQuestionIndex(0);
        setUserAnswers([]);
        setTimeLeft(category.duration * 60); // Convert minutes to seconds
        setQuizStarted(true);
        setCurrentView('quiz');
      } else {
        // For predefined categories, call the API
        const { data } = await axios.post("http://localhost:8000/skill-assessment/generate", {
          category: category.id,
          difficulty: category.difficulty.toLowerCase(),
          num_questions: category.questions
        });
        
        setCurrentQuiz(data);
        setSelectedCategory(category);
        setCurrentQuestionIndex(0);
        setUserAnswers([]);
        setTimeLeft(category.duration.split(' ')[0] * 60); // Convert minutes to seconds
        setQuizStarted(true);
        setCurrentView('quiz');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to generate quiz. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleAnswerSelect = (answerIndex) => {
    const newAnswers = [...userAnswers];
    newAnswers[currentQuestionIndex] = answerIndex;
    setUserAnswers(newAnswers);
  };

  const goToNextQuestion = () => {
    if (currentQuestionIndex < currentQuiz.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      handleQuizComplete();
    }
  };

  const goToPreviousQuestion = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleQuizComplete = async () => {
    setQuizStarted(false);
    setLoading(true);
    
    try {
      // For custom quizzes, we need to handle evaluation locally since the API might not support them
      if (selectedCategory.isCustom) {
        // Calculate score
        const correctCount = userAnswers.reduce((count, answer, index) => {
          if (index < currentQuiz.questions.length && answer === currentQuiz.questions[index].correct_answer) {
            return count + 1;
          }
          return count;
        }, 0);
        
        const totalQuestions = currentQuiz.questions.length;
        const score = Math.round((correctCount / totalQuestions) * 100);
        
        // Determine performance level
        let performanceLevel = "beginner";
        if (score >= 90) performanceLevel = "expert";
        else if (score >= 75) performanceLevel = "advanced";
        else if (score >= 60) performanceLevel = "intermediate";
        
        // Create custom evaluation result
        const evaluationResult = {
          score,
          correct_count: correctCount,
          total_questions: totalQuestions,
          performance_level: performanceLevel,
          strengths: ["Knowledge in selected topics", "Quiz completion"],
          weaknesses: ["Areas with incorrect answers", "Topics needing more review"],
          recommendations: [
            "Review questions you got wrong",
            "Create more quizzes on challenging topics",
            "Try standard assessments for comparison"
          ],
          detailed_feedback: `You completed the custom quiz "${selectedCategory.title}" with a score of ${score}%. Keep practicing to improve your knowledge in this area.`
        };
        
        // Remove the completed quiz since it should expire after being taken
        if (selectedCategory.id) {
          // Filter out the completed quiz from customQuizzes state
          const updatedQuizzes = customQuizzes.filter(quiz => quiz.id !== selectedCategory.id);
          setCustomQuizzes(updatedQuizzes);
          
          // Update localStorage
          localStorage.setItem('customQuizzes', JSON.stringify(updatedQuizzes));
          
          // Try to remove from backend if API is available
          try {
            await axios.delete(`http://localhost:8000/custom-quiz/delete/${selectedCategory.id}`);
          } catch (err) {
            console.log("Using only localStorage for removing completed quiz");
          }
        }
        
        setResults(evaluationResult);
        setCurrentView('results');
      } else {
        // For standard categories, use the API
        const { data } = await axios.post("http://localhost:8000/skill-assessment/evaluate", {
          category: selectedCategory.id,
          difficulty: selectedCategory.difficulty.toLowerCase(),
          answers: userAnswers,
          questions: currentQuiz.questions
        });
        
        setResults(data);
        setCurrentView('results');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to evaluate quiz. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetAssessment = () => {
    setCurrentView('categories');
    setSelectedCategory(null);
    setCurrentQuiz(null);
    setCurrentQuestionIndex(0);
    setUserAnswers([]);
    setTimeLeft(0);
    setQuizStarted(false);
    setResults(null);
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty.toLowerCase()) {
      case 'beginner': return '#2ecc71';
      case 'intermediate': return '#f39c12';
      case 'advanced': return '#e74c3c';
      default: return '#95a5a6';
    }
  };

  const getScoreColor = (percentage) => {
    if (percentage >= 80) return '#2ecc71';
    if (percentage >= 60) return '#f39c12';
    return '#e74c3c';
  };

  const formatPerformanceLevel = (level) => {
    switch (level.toLowerCase()) {
      case 'expert': return { label: 'Expert Level', icon: '🏆', color: '#2ecc71', description: 'Outstanding mastery of the subject' };
      case 'advanced': return { label: 'Advanced', icon: '⭐', color: '#3498db', description: 'Strong knowledge and skills' };
      case 'intermediate': return { label: 'Intermediate', icon: '📈', color: '#f39c12', description: 'Good understanding with room to grow' };
      case 'beginner': return { label: 'Beginner', icon: '🌱', color: '#e74c3c', description: 'Great starting point for learning' };
      default: return { label: 'Assessment Complete', icon: '✅', color: '#6c757d', description: 'Results available' };
    }
  };

  const formatScoreMessage = (score) => {
    if (score >= 90) return "Outstanding performance! You demonstrate expert-level knowledge.";
    if (score >= 75) return "Great work! You have strong knowledge in this area.";
    if (score >= 60) return "Good job! You have solid understanding with room for growth.";
    return "Keep learning! There's great potential for improvement.";
  };

  const formatStrengthsAndWeaknesses = (items) => {
    return items.map(item => item.charAt(0).toUpperCase() + item.slice(1).toLowerCase());
  };

  /* ---------- text formatting utilities ---------- */
  const cleanText = (text) => {
    if (!text) return '';
    
    // Remove common LLM artifacts and formatting
    return text
      // Remove markdown formatting
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markers
      .replace(/\*(.*?)\*/g, '$1')     // Remove italic markers
      .replace(/`(.*?)`/g, '$1')       // Remove code markers
      .replace(/#{1,6}\s+/g, '')       // Remove heading markers
      .replace(/^\s*[-*+]\s+/gm, '')   // Remove bullet points
      .replace(/^\s*\d+\.\s+/gm, '')   // Remove numbered lists
      // Remove common prefixes and artifacts
      .replace(/^(Answer:|Question:|Explanation:|Note:|STRENGTHS:|WEAKNESSES:|RECOMMENDATIONS:|DETAILED_FEEDBACK:)/gi, '')
      .replace(/^[-•*]\s*/gm, '')      // Remove bullet points
      .replace(/^\d+\.\s*/gm, '')      // Remove numbers
      .replace(/^[A-D]\)\s*/gm, '')    // Remove option letters
      // Clean up punctuation and spacing
      .replace(/\s+/g, ' ')            // Normalize whitespace
      .replace(/([.!?])\s*([.!?])/g, '$1') // Remove duplicate punctuation
      .trim();
  };

  const formatQuestionText = (text) => {
    const cleaned = cleanText(text);
    // Ensure it ends with proper punctuation
    if (cleaned && !cleaned.match(/[.!?]$/)) {
      return cleaned + '?';
    }
    return cleaned;
  };

  const formatOptionText = (text) => {
    const cleaned = cleanText(text);
    // Remove any leading option letters or numbers that might have been missed
    return cleaned.replace(/^[A-D][\)\.]\s*/i, '').replace(/^\d+[\)\.]\s*/, '');
  };

  const formatFeedbackText = (text) => {
    const cleaned = cleanText(text);
    // Ensure proper sentence structure
    if (cleaned && !cleaned.match(/[.!?]$/)) {
      return cleaned + '.';
    }
    return cleaned;
  };

  const formatListItem = (text) => {
    const cleaned = cleanText(text);
    // Capitalize first letter and ensure no trailing punctuation for list items
    if (cleaned) {
      const formatted = cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
      return formatted.replace(/[.!?]+$/, ''); // Remove trailing punctuation
    }
    return cleaned;
  };

  // Create Quiz View
  if (currentView === 'create-quiz') {
    return (
      <CreateQuiz 
        onBack={() => setCurrentView('categories')} 
        onQuizCreated={handleQuizCreated}
      />
    );
  }
  
  // Categories View
  if (currentView === 'categories') {
    return (
      <div className="skill-assessment-wrapper">
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
            <span>Skill Assessment Center</span>
          </div>
          <button 
            className="create-quiz-btn" 
            onClick={() => setCurrentView('create-quiz')}
            title="Create Your Own Quiz"
          >
            <Plus size={16} />
            Create Quiz
          </button>
        </header>

        <div className="assessment-container">
          <div className="assessment-intro">
            <div className="intro-content">
              <Trophy className="intro-icon" size={48} />
              <h1>Data Professional Skills Assessment</h1>
              <p>
                Test your expertise across various data science and engineering domains. 
                Choose a category below to start your assessment and receive detailed feedback on your performance.
              </p>
            </div>
          </div>

          <h2 className="section-title">Standard Assessments</h2>
          <div className="categories-grid">
            {skillCategories.map((category) => (
              <div 
                key={category.id} 
                className="category-card"
                style={{ borderColor: category.color }}
              >
                <div className="category-header">
                  <div className="category-icon-wrapper" style={{ backgroundColor: category.color }}>
                    {category.icon}
                  </div>
                  <div className="category-info">
                    <h3>{category.title}</h3>
                    <div className="category-meta">
                      <span 
                        className="difficulty-badge"
                        style={{ backgroundColor: getDifficultyColor(category.difficulty) }}
                      >
                        {category.difficulty}
                      </span>
                      <span className="duration-info">
                        <Clock size={14} />
                        {category.duration}
                      </span>
                      <span className="questions-info">
                        <Target size={14} />
                        {category.questions} questions
                      </span>
                    </div>
                  </div>
                </div>
                <p className="category-description">{category.description}</p>
                <button 
                  className="start-quiz-btn"
                  onClick={() => startQuiz(category)}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <RotateCw size={16} className="spinning" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Play size={16} />
                      Start Assessment
                    </>
                  )}
                </button>
              </div>
            ))}
          </div>
          
          {customQuizzes.length > 0 && (
            <>
              <h2 className="section-title">Your Custom Quizzes</h2>
              <div className="quiz-info-banner">
                <Clock size={16} />
                <span>Note: Custom quizzes will automatically expire after completion</span>
              </div>
              <div className="categories-grid">
                {customQuizzes.map(quiz => (
                  <div key={quiz.id} className="category-card custom-quiz-card">
                    <div className="category-header">
                      <div className="category-icon-wrapper" style={{ background: '#3498db' }}>
                        <Bookmark className="category-icon" />
                      </div>
                      <div className="category-info">
                        <h3>{quiz.title}</h3>
                        <div className="category-meta">
                          <span 
                            className="difficulty-badge" 
                            style={{ background: getDifficultyColor(quiz.difficulty) }}
                          >
                            {quiz.difficulty.charAt(0).toUpperCase() + quiz.difficulty.slice(1)}
                          </span>
                          <span className="duration-info">
                            <Clock size={14} />
                            {quiz.duration} mins
                          </span>
                          <span className="questions-info">
                            <FileText size={14} />
                            {quiz.questions.length} Questions
                          </span>
                        </div>
                      </div>
                      <button 
                        className="delete-quiz-btn"
                        onClick={(e) => deleteQuiz(quiz.id, e)}
                        aria-label="Delete quiz"
                        title="Delete quiz"
                      >
                        <XCircle size={18} />
                      </button>
                    </div>
                    <p className="category-description">{quiz.description || "Custom quiz created by you."}</p>
                    <button 
                      className="start-quiz-btn custom-quiz-btn" 
                      onClick={() => startQuiz(quiz)}
                      disabled={loading}
                    >
                      {loading ? (
                        <>
                          <RotateCw className="spinning" size={16} />
                          Loading...
                        </>
                      ) : (
                        <>
                          <Play size={16} />
                          Start Quiz
                        </>
                      )}
                    </button>
                  </div>
                ))}
                
                <div className="category-card add-quiz-card" onClick={() => setCurrentView('create-quiz')}>
                  <div className="add-quiz-content">
                    <Plus size={48} />
                    <h3>Create New Quiz</h3>
                    <p>Create your own custom quiz with personalized questions and answers</p>
                  </div>
                </div>
              </div>
            </>
          )}
          
          {customQuizzes.length === 0 && (
            <div className="create-first-quiz">
              <h2>Create Your Own Quiz</h2>
              <p>Want to make a custom quiz? It's easy to create your own questions and share with others.</p>
              <button 
                className="create-quiz-button" 
                onClick={() => setCurrentView('create-quiz')}
              >
                <Plus size={18} />
                Create My First Quiz
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Quiz View
  if (currentView === 'quiz' && currentQuiz) {
    const currentQuestion = currentQuiz.questions[currentQuestionIndex];
    const progress = ((currentQuestionIndex + 1) / currentQuiz.questions.length) * 100;

    return (
      <div className="skill-assessment-wrapper">
        <header className="quiz-header">
          <div className="quiz-info">
            <h3>{selectedCategory.title}</h3>
            <div className="quiz-meta">
              <span>Question {currentQuestionIndex + 1} of {currentQuiz.questions.length}</span>
              <span className="timer">
                <Clock size={16} />
                {formatTime(timeLeft)}
              </span>
            </div>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
          </div>
        </header>

        <div className="quiz-container">
          <div className="question-card">
            <div className="question-header">
              <h2>Question {currentQuestionIndex + 1}</h2>
              {currentQuestion.type === 'code' && (
                <span className="question-type-badge">
                  <Code size={14} />
                  Code Challenge
                </span>
              )}
            </div>
            
            <div className="question-content">
              <p className="preview_question-text">{formatQuestionText(currentQuestion.question)}</p>
              
              {currentQuestion.code_snippet && (
                <div className="code-snippet">
                  <pre><code>{cleanText(currentQuestion.code_snippet)}</code></pre>
                </div>
              )}
            </div>

            <div className="answer-options">
              {currentQuestion.options.map((option, index) => (
                <button
                  key={index}
                  className={`answer-option ${userAnswers[currentQuestionIndex] === index ? 'selected' : ''}`}
                  onClick={() => handleAnswerSelect(index)}
                >
                  <span className="option-letter">{String.fromCharCode(65 + index)}</span>
                  <span className="option-text">{formatOptionText(option)}</span>
                  {userAnswers[currentQuestionIndex] === index && (
                    <CheckCircle size={20} className="selected-icon" />
                  )}
                </button>
              ))}
            </div>

            <div className="quiz-navigation">
              <button 
                className="nav-btn secondary"
                onClick={goToPreviousQuestion}
                disabled={currentQuestionIndex === 0}
              >
                <ChevronLeft size={16} />
                Previous
              </button>
              
              <span className="question-counter">
                {currentQuestionIndex + 1} / {currentQuiz.questions.length}
              </span>
              
              <button 
                className="nav-btn primary"
                onClick={goToNextQuestion}
                disabled={userAnswers[currentQuestionIndex] === undefined}
              >
                {currentQuestionIndex === currentQuiz.questions.length - 1 ? (
                  <>
                    <Award size={16} />
                    Complete Quiz
                  </>
                ) : (
                  <>
                    Next
                    <ChevronRight size={16} />
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Results View
  if (currentView === 'results' && results) {
    return (
      <div className="skill-assessment-wrapper">
        <header className="chat-header">
          <div className="header-left">
            <button 
              className="back-btn" 
              onClick={resetAssessment}
              title="Back to Categories"
            >
              <ArrowLeft size={20} />
            </button>
            <span>Assessment Results</span>
          </div>
        </header>

        <div className="results-container">
          <div className="results-card">
            {/* Enhanced Results Header */}
            <div className="results-header">
              <div className="score-circle" style={{ borderColor: getScoreColor(results.score) }}>
                <span className="score-percentage">{results.score}%</span>
                <span className="score-label">Score</span>
              </div>
              <div className="results-info">
                <h2>{selectedCategory.title} Assessment</h2>
                <p className="score-message">{formatScoreMessage(results.score)}</p>
                <div className="results-stats">
                  <div className="stat">
                    <CheckCircle size={16} className="correct-icon" />
                    <span><strong>{results.correct_count}</strong> out of <strong>{results.total_questions}</strong> correct</span>
                  </div>
                  <div className="stat">
                    <Target size={16} />
                    <span>Accuracy: <strong>{Math.round((results.correct_count / results.total_questions) * 100)}%</strong></span>
                  </div>
                  <div className="stat">
                    <Clock size={16} />
                    <span>Assessment completed</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Enhanced Performance Level */}
            <div className="performance-level">
              <h3>Performance Level</h3>
              <div className="level-indicator">
                <div className="level-info">
                  <span className="level-icon">{formatPerformanceLevel(results.performance_level).icon}</span>
                  <div className="level-details">
                    <span className="level-label" style={{ color: formatPerformanceLevel(results.performance_level).color }}>
                      {formatPerformanceLevel(results.performance_level).label}
                    </span>
                    <span className="level-description">
                      {formatPerformanceLevel(results.performance_level).description}
                    </span>
                  </div>
                </div>
                <div className="level-bar">
                  <div 
                    className="level-fill" 
                    style={{ 
                      width: `${results.score}%`,
                      backgroundColor: getScoreColor(results.score)
                    }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Enhanced Feedback Section */}
            {results.detailed_feedback && (
              <div className="feedback-section">
                <h3>📝 Detailed Feedback</h3>
                <div className="feedback-content">
                  <p>{formatFeedbackText(results.detailed_feedback)}</p>
                </div>
              </div>
            )}

            {/* Enhanced Strengths and Weaknesses */}
            <div className="strengths-weaknesses">
              <div className="strengths">
                <h4><Star size={16} /> 💪 Your Strengths</h4>
                {results.strengths && results.strengths.length > 0 ? (
                  <ul>
                    {results.strengths.map((strength, index) => (
                      <li key={index}>✅ {formatListItem(strength)}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="no-items">Keep working to identify your strengths!</p>
                )}
              </div>
              <div className="weaknesses">
                <h4><Target size={16} /> 🎯 Areas for Growth</h4>
                {results.weaknesses && results.weaknesses.length > 0 ? (
                  <ul>
                    {results.weaknesses.map((area, index) => (
                      <li key={index}>🔄 {formatListItem(area)}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="no-items">Great job! No major areas for improvement identified.</p>
                )}
              </div>
            </div>

            {/* Enhanced Recommendations */}
            <div className="recommendations">
              <h3>🚀 Recommended Next Steps</h3>
              {results.recommendations && results.recommendations.length > 0 ? (
                <div className="recommendations-list">
                  {results.recommendations.map((recommendation, index) => (
                    <div key={index} className="recommendation-item">
                      <span className="recommendation-number">{index + 1}</span>
                      <span className="recommendation-text">{formatFeedbackText(recommendation)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="no-items">Continue your learning journey at your own pace!</p>
              )}
            </div>

            {/* Enhanced Action Buttons */}
            <div className="results-actions">
              <button 
                className="action-btn secondary"
                onClick={resetAssessment}
                title="Choose a different category or retake this assessment"
              >
                <RotateCcw size={16} />
                Take Another Assessment
              </button>
              <button 
                className="action-btn primary"
                onClick={() => {
                  const shareText = `I scored ${results.score}% on the ${selectedCategory.title} assessment! 🎉`;
                  if (navigator.share) {
                    navigator.share({ title: 'Assessment Results', text: shareText });
                  } else {
                    navigator.clipboard.writeText(shareText);
                    alert('Results copied to clipboard!');
                  }
                }}
                title="Share your achievement"
              >
                <Share2 size={16} />
                Share Results
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
