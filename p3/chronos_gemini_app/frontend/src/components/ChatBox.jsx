import { useState, useRef, useEffect } from "react";
import { Send, Zap, BarChart3, FileText } from "lucide-react";

const quick = [
  { icon: Zap, text: "Forecast next 8 steps", prompt: "Forecast the next 8 steps with the selected model." },
  { icon: BarChart3, text: "Show metrics", prompt: "Give me WQL and MASE for the last forecast." },
  { icon: FileText, text: "Explain model", prompt: "Tell me about the Chronos model family." },
];

export default function ChatBox({ geminiKey, sessionId, dataset, model, setForecast, setEvaluation }) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hello! Upload a CSV and ask me anything about forecasting.", time: new Date() },
  ]);
  const endRef = useRef(null);

  const send = async (text) => {
    if (!geminiKey) return alert("Set Gemini API key first");
    const userMsg = { role: "user", content: text, time: new Date() };
    setMessages((m) => [...m, userMsg]);
    setInput("");

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId, gemini_key: geminiKey }),
    });
    const data = await res.json();
    setMessages((m) => [...m, { role: "assistant", content: data.response, time: new Date() }]);

    // Auto‑trigger forecast if user asks for it
    if (text.toLowerCase().includes("forecast")) {
      const fRes = await fetch("/api/forecast", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, model_name: model, prediction_length: 8 }),
      });
      const f = await fRes.json();
      setForecast(f.forecast);
      setEvaluation(f.evaluation);
    }
  };

  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-xs px-4 py-2 rounded-lg ${
                m.role === "user" ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-800"
              }`}
            >
              <p className="whitespace-pre-wrap">{m.content}</p>
              <span className="text-xs opacity-70">{m.time.toLocaleTimeString()}</span>
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      {/* Quick actions */}
      {dataset && (
        <div className="flex gap-2 p-2 border-t">
          {quick.map((q, i) => (
            <button
              key={i}
              onClick={() => send(q.prompt)}
              className="flex items-center gap-1 px-3 py-1 text-sm bg-indigo-50 text-indigo-700 rounded-full hover:bg-indigo-100"
            >
              <q.icon className="w-4 h-4" />
              {q.text}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <form
        onSubmit={(e) => { e.preventDefault(); send(input); }}
        className="flex items-center p-3 border-t"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about forecasting, metrics, or code..."
          className="flex-1 px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <button type="submit" className="ml-2 p-2 text-indigo-600 hover:bg-indigo-50 rounded-lg">
          <Send className="w-5 h-5" />
        </button>
      </form>
    </div>
  );
}