import { ArrowLeft } from "lucide-react";

export default function PromptEnhancerTest({ onBackToTools }) {
  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)', 
      color: 'white',
      padding: '2rem'
    }}>
      <button
        onClick={onBackToTools}
        style={{ 
          background: 'rgba(255, 255, 255, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          color: 'white',
          padding: '0.5rem 1rem',
          borderRadius: '8px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}
      >
        <ArrowLeft size={18} />
        Back to Tools
      </button>
      
      <h1 style={{ marginTop: '2rem', fontSize: '2rem' }}>
        Test Prompt Enhancer
      </h1>
      
      <p style={{ marginTop: '1rem' }}>
        This is a test version to check if the component renders.
      </p>
      
      <div style={{ 
        background: 'rgba(255, 255, 255, 0.1)',
        padding: '1rem',
        borderRadius: '8px',
        marginTop: '2rem'
      }}>
        <textarea 
          placeholder="Test input..."
          style={{
            width: '100%',
            background: 'white',
            color: 'black',
            padding: '0.75rem',
            borderRadius: '8px',
            border: 'none',
            minHeight: '40px'
          }}
        />
      </div>
    </div>
  );
}
