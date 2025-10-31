import { useState } from "react";
import ChatBox from "./components/ChatBox";
import UploadZone from "./components/UploadZone";
import ForecastChart from "./components/ForecastChart";
import MetricsPanel from "./components/MetricsPanel";
import ModelSelector from "./components/ModelSelector";
import { Settings } from "lucide-react";

export default function App() {
  const [geminiKey, setGeminiKey] = useState("");
  const [sessionId] = useState("default");
  const [dataset, setDataset] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const [model, setModel] = useState("amazon/chronos-t5-base");

  return (
    <div className="flex h-screen bg-gradient-to-br from-indigo-50 to-purple-100">
      {/* Sidebar */}
      <aside className="w-80 bg-white shadow-xl p-6 flex flex-col gap-6 overflow-y-auto">
        <div className="flex items-center gap-3">
          <Settings className="w-6 h-6 text-indigo-600" />
          <h2 className="text-xl font-semibold">Gemini API Key</h2>
        </div>
        <input
          type="password"
          placeholder="AIza..."
          value={geminiKey}
          onChange={(e) => setGeminiKey(e.target.value)}
          className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <ModelSelector model={model} setModel={setModel} />
        <UploadZone setDataset={setDataset} sessionId={sessionId} />
        {evaluation && <MetricsPanel eval={evaluation} />}
      </aside>

      {/* Main area */}
      <main className="flex-1 flex flex-col">
        <header className="bg-white shadow-sm p-4 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-800">Chronos AI Assistant</h1>
          <p className="text-sm text-gray-600">Powered by Gemini &amp; Chronos T5</p>
        </header>

        <section className="flex-1 overflow-y-auto p-6">
          {forecast && <ForecastChart data={forecast} history={dataset?.history} />}
          <ChatBox
            geminiKey={geminiKey}
            sessionId={sessionId}
            dataset={dataset}
            model={model}
            setForecast={setForecast}
            setEvaluation={setEvaluation}
          />
        </section>
      </main>
    </div>
  );
}