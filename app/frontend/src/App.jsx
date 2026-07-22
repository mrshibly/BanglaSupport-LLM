import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Database, Wrench, MessageSquare } from 'lucide-react';

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: 'আসসালামু আলাইকুম! ই-কমার্স গ্রাহক সেবা সহায়তায় আপনাকে স্বাগতম। আমি কিভাবে সাহায্য করতে পারি?',
      mode: 'direct'
    }
  ]);
  const [input, setInput] = useState('');
  const [mode, setMode] = useState('direct'); // direct | rag | agent
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: input,
      mode
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, mode })
      });

      const data = await response.json();

      const botMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: data.response,
        mode: data.mode,
        retrievedContext: data.retrieved_context,
        toolCalled: data.tool_called,
        toolArgs: data.tool_args
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'bot',
          text: 'দুঃখিত, সার্ভারের সাথে সংযোগ স্থাপন করা সম্ভব হয়নি। অনুগ্রহ করে backend চালু করুন।',
          mode
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="chat-title">
          <h2>বাংলা ই-কমার্স এআই সাপোর্ট</h2>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Qwen3-8B Fine-tuned LLM</span>
        </div>

        <div className="mode-selectors">
          <button
            className={`mode-btn ${mode === 'direct' ? 'active' : ''}`}
            onClick={() => setMode('direct')}
          >
            <MessageSquare size={14} style={{ display: 'inline', marginRight: 4 }} /> Direct LLM
          </button>
          <button
            className={`mode-btn ${mode === 'rag' ? 'active' : ''}`}
            onClick={() => setMode('rag')}
          >
            <Database size={14} style={{ display: 'inline', marginRight: 4 }} /> RAG Mode
          </button>
          <button
            className={`mode-btn ${mode === 'agent' ? 'active' : ''}`}
            onClick={() => setMode('agent')}
          >
            <Wrench size={14} style={{ display: 'inline', marginRight: 4 }} /> Agentic Tools
          </button>
        </div>
      </header>

      <main className="messages-list">
        {messages.map(msg => (
          <div key={msg.id} className={`message-item ${msg.sender}`}>
            <div className={`avatar ${msg.sender}`}>
              {msg.sender === 'user' ? <User size={18} /> : <Bot size={18} />}
            </div>
            <div>
              <div className="bubble">{msg.text}</div>

              {msg.retrievedContext && msg.retrievedContext.length > 0 && (
                <div className="context-tag">
                  <strong>📚 RAG রিট্রিভড তথ্য:</strong>
                  {msg.retrievedContext.map((c, i) => (
                    <div key={i} style={{ marginTop: 2 }}>• {c.slice(0, 100)}...</div>
                  ))}
                </div>
              )}

              {msg.toolCalled && (
                <div className="tool-tag">
                  <strong>⚡ এক্সিকিউটেড টুল:</strong> {msg.toolCalled}({JSON.stringify(msg.toolArgs)})
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="message-item bot">
            <div className="avatar bot"><Bot size={18} /></div>
            <div className="bubble" style={{ color: '#94a3b8' }}>টাইপ করা হচ্ছে...</div>
          </div>
        )}
        <div ref={chatEndRef} />
      </main>

      <footer className="input-area">
        <input
          type="text"
          placeholder="আপনার প্রশ্ন বাংলায় লিখুন... (যেমন: আমার অর্ডার BD1001 কোথায়?)"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
        />
        <button className="send-btn" onClick={handleSend} disabled={loading}>
          <Send size={16} /> পাঠান
        </button>
      </footer>
    </div>
  );
}
