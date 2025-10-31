import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, TrendingUp, BarChart3, Download, Loader2, AlertCircle, Info } from 'lucide-react';

const ChronosGeminiChat = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '👋 Hello! I\'m your Chronos AI assistant powered by Gemini. Upload your time series dataset and ask me anything about forecasting, data analysis, or model configuration!',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [dataset, setDataset] = useState(null);
  const [datasetStats, setDatasetStats] = useState(null);
  const [geminiKey, setGeminiKey] = useState('');
  const [showKeyInput, setShowKeyInput] = useState(true);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Parse CSV data
  const parseCSV = (text) => {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    const data = lines.slice(1).map(line => {
      const values = line.split(',');
      const row = {};
      headers.forEach((header, i) => {
        row[header] = values[i]?.trim();
      });
      return row;
    });

    return { headers, data };
  };

  // Analyze dataset
  const analyzeDataset = (parsedData) => {
    const { headers, data } = parsedData;
    
    // Find numeric columns
    const numericCols = headers.filter(header => {
      const sample = data[0][header];
      return !isNaN(parseFloat(sample));
    });

    // Calculate basic stats
    const stats = {
      totalRows: data.length,
      totalColumns: headers.length,
      headers: headers,
      numericColumns: numericCols,
      dateColumns: headers.filter(h => 
        h.toLowerCase().includes('date') || 
        h.toLowerCase().includes('time') ||
        h.toLowerCase().includes('timestamp')
      ),
      targetColumns: headers.filter(h => 
        h.toLowerCase().includes('target') || 
        h.toLowerCase().includes('value') ||
        h.toLowerCase().includes('sales')
      ),
      preview: data.slice(0, 5)
    };

    // Calculate statistics for numeric columns
    if (numericCols.length > 0 && stats.targetColumns.length > 0) {
      const targetCol = stats.targetColumns[0];
      const values = data.map(row => parseFloat(row[targetCol])).filter(v => !isNaN(v));
      
      if (values.length > 0) {
        values.sort((a, b) => a - b);
        stats.targetStats = {
          mean: values.reduce((a, b) => a + b, 0) / values.length,
          median: values[Math.floor(values.length / 2)],
          min: values[0],
          max: values[values.length - 1],
          std: Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - (values.reduce((a, b) => a + b, 0) / values.length), 2), 0) / values.length)
        };
      }
    }

    return stats;
  };

  // Handle file upload
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const parsed = parseCSV(text);
        const stats = analyzeDataset(parsed);
        
        setDataset(parsed);
        setDatasetStats(stats);

        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `📊 Dataset loaded successfully!\n\n**Summary:**\n- Total rows: ${stats.totalRows}\n- Total columns: ${stats.totalColumns}\n- Date columns: ${stats.dateColumns.join(', ') || 'None detected'}\n- Target columns: ${stats.targetColumns.join(', ') || 'None detected'}\n- Numeric columns: ${stats.numericColumns.join(', ')}\n\n${stats.targetStats ? `**Target Statistics:**\n- Mean: ${stats.targetStats.mean.toFixed(2)}\n- Median: ${stats.targetStats.median.toFixed(2)}\n- Min: ${stats.targetStats.min.toFixed(2)}\n- Max: ${stats.targetStats.max.toFixed(2)}\n- Std Dev: ${stats.targetStats.std.toFixed(2)}` : ''}\n\nYou can now ask me questions about your data!`,
          timestamp: new Date()
        }]);
      } catch (error) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `❌ Error parsing file: ${error.message}. Please ensure it's a valid CSV file.`,
          timestamp: new Date()
        }]);
      }
    };
    reader.readAsText(file);
  };

  // Build context for Gemini
  const buildContext = () => {
    if (!dataset || !datasetStats) return '';

    return `
DATASET CONTEXT:
- You are analyzing a time series dataset
- Total rows: ${datasetStats.totalRows}
- Columns: ${datasetStats.headers.join(', ')}
- Date/Time columns: ${datasetStats.dateColumns.join(', ') || 'None'}
- Target columns: ${datasetStats.targetColumns.join(', ') || 'None'}
- Numeric columns: ${datasetStats.numericColumns.join(', ')}

${datasetStats.targetStats ? `
TARGET VARIABLE STATISTICS:
- Mean: ${datasetStats.targetStats.mean.toFixed(4)}
- Median: ${datasetStats.targetStats.median.toFixed(4)}
- Min: ${datasetStats.targetStats.min.toFixed(4)}
- Max: ${datasetStats.targetStats.max.toFixed(4)}
- Standard Deviation: ${datasetStats.targetStats.std.toFixed(4)}
- Coefficient of Variation: ${(datasetStats.targetStats.std / datasetStats.targetStats.mean).toFixed(4)}
` : ''}

SAMPLE DATA (first 3 rows):
${JSON.stringify(datasetStats.preview.slice(0, 3), null, 2)}

CHRONOS MODEL CAPABILITIES:
- Chronos is a pretrained time series forecasting model based on T5 architecture
- Available models: chronos-t5-tiny, mini, small, base, large
- Supports zero-shot forecasting (no training needed)
- Can predict multiple horizons
- Generates probabilistic forecasts with quantiles
- Works best with context length 64-512 timesteps
- Recommended for: sales, demand, finance, energy, weather forecasting

YOUR ROLE:
- Analyze the user's dataset and provide insights
- Suggest optimal Chronos configuration (model size, context length, prediction horizon)
- Recommend preprocessing steps
- Explain forecasting strategies
- Help interpret results
- Provide Python code snippets when helpful
- Be conversational and helpful

Answer the user's question based on their dataset context.
`;
  };

  // Call Gemini API
  const callGeminiAPI = async (userMessage) => {
    if (!geminiKey) {
      return 'Please set your Gemini API key first using the settings button.';
    }

    const context = buildContext();
    const conversationHistory = messages.slice(-5).map(msg => ({
      role: msg.role === 'assistant' ? 'model' : 'user',
      parts: [{ text: msg.content }]
    }));

    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${geminiKey}`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            contents: [
              { role: 'user', parts: [{ text: context }] },
              ...conversationHistory,
              { role: 'user', parts: [{ text: userMessage }] }
            ],
            generationConfig: {
              temperature: 0.7,
              maxOutputTokens: 2048,
            }
          })
        }
      );

      const data = await response.json();
      
      if (data.candidates && data.candidates[0]?.content?.parts?.[0]?.text) {
        return data.candidates[0].content.parts[0].text;
      } else if (data.error) {
        return `Error: ${data.error.message}`;
      } else {
        return 'Unable to get response from Gemini. Please check your API key.';
      }
    } catch (error) {
      console.error('Gemini API error:', error);
      return `Error calling Gemini API: ${error.message}`;
    }
  };

  // Handle send message
  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await callGeminiAPI(input);
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response,
        timestamp: new Date()
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message}`,
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Quick action buttons
  const quickActions = [
    { 
      label: 'Analyze my data', 
      icon: BarChart3,
      prompt: 'Can you analyze my dataset and tell me what patterns you see? What forecasting approach would work best?'
    },
    { 
      label: 'Suggest config', 
      icon: TrendingUp,
      prompt: 'What Chronos model configuration would you recommend for my dataset? Include model size, context length, and prediction horizon.'
    },
    { 
      label: 'Code example', 
      icon: Download,
      prompt: 'Can you give me a complete Python code example for forecasting with Chronos on my dataset?'
    }
  ];

  const handleQuickAction = (prompt) => {
    setInput(prompt);
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-lg border-b border-indigo-200">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-indigo-600 p-2 rounded-lg">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-800">Chronos AI Assistant</h1>
                <p className="text-sm text-gray-600">Powered by Gemini & Chronos T5</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              {dataset && (
                <div className="bg-green-100 px-4 py-2 rounded-lg">
                  <p className="text-sm font-medium text-green-800">
                    📊 Dataset loaded: {datasetStats.totalRows} rows
                  </p>
                </div>
              )}
              
              <button
                onClick={() => setShowKeyInput(!showKeyInput)}
                className="bg-gray-100 hover:bg-gray-200 p-2 rounded-lg transition-colors"
                title="Settings"
              >
                <Info className="w-5 h-5 text-gray-700" />
              </button>
            </div>
          </div>

          {/* API Key Input */}
          {showKeyInput && (
            <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex items-start space-x-2">
                <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-yellow-800 mb-2">
                    Enter your Gemini API Key
                  </p>
                  <input
                    type="password"
                    value={geminiKey}
                    onChange={(e) => setGeminiKey(e.target.value)}
                    placeholder="AIza..."
                    className="w-full px-3 py-2 border border-yellow-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-500"
                  />
                  <p className="text-xs text-yellow-700 mt-2">
                    Get your free API key at: <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="underline">Google AI Studio</a>
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 overflow-hidden flex flex-col max-w-6xl mx-auto w-full">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-3xl rounded-2xl px-6 py-4 shadow-md ${
                  message.role === 'user'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-white text-gray-800 border border-gray-200'
                }`}
              >
                <div className="whitespace-pre-wrap break-words">
                  {message.content.split('\n').map((line, i) => {
                    // Handle code blocks
                    if (line.startsWith('```')) {
                      return null;
                    }
                    // Handle headers
                    if (line.startsWith('**') && line.endsWith('**')) {
                      return (
                        <div key={i} className="font-bold text-lg mt-3 mb-2">
                          {line.replace(/\*\*/g, '')}
                        </div>
                      );
                    }
                    // Handle bullet points
                    if (line.trim().startsWith('- ')) {
                      return (
                        <div key={i} className="ml-4 mb-1">
                          • {line.replace(/^- /, '')}
                        </div>
                      );
                    }
                    // Regular text
                    return line ? <div key={i} className="mb-1">{line}</div> : <br key={i} />;
                  })}
                </div>
                <p className="text-xs opacity-60 mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white rounded-2xl px-6 py-4 shadow-md border border-gray-200">
                <div className="flex items-center space-x-2">
                  <Loader2 className="w-5 h-5 animate-spin text-indigo-600" />
                  <span className="text-gray-600">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Actions */}
        {!isLoading && dataset && (
          <div className="px-6 py-2">
            <div className="flex flex-wrap gap-2">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickAction(action.prompt)}
                  className="flex items-center space-x-2 bg-white hover:bg-indigo-50 border border-indigo-200 px-4 py-2 rounded-full text-sm font-medium text-indigo-700 transition-colors shadow-sm"
                >
                  <action.icon className="w-4 h-4" />
                  <span>{action.label}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <div className="flex items-end space-x-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
            />
            
            <button
              onClick={() => fileInputRef.current?.click()}
              className="bg-gray-100 hover:bg-gray-200 p-3 rounded-xl transition-colors"
              title="Upload CSV dataset"
            >
              <Upload className="w-6 h-6 text-gray-700" />
            </button>

            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder={dataset ? "Ask me anything about your data or Chronos forecasting..." : "Upload a CSV dataset first to get started..."}
                disabled={!geminiKey || isLoading}
                className="w-full px-4 py-3 pr-12 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 resize-none"
                rows={1}
                style={{ minHeight: '52px', maxHeight: '120px' }}
              />
            </div>

            <button
              onClick={handleSendMessage}
              disabled={!input.trim() || !geminiKey || isLoading}
              className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed p-3 rounded-xl transition-colors shadow-lg"
            >
              {isLoading ? (
                <Loader2 className="w-6 h-6 text-white animate-spin" />
              ) : (
                <Send className="w-6 h-6 text-white" />
              )}
            </button>
          </div>
          
          <p className="text-xs text-gray-500 mt-2 text-center">
            Press Enter to send • Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChronosGeminiChat;