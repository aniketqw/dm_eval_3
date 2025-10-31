import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function ForecastChart({ data, history }) {
  const chartData = [
    ...(history?.map((v, i) => ({ step: i - history.length, value: v, type: "history" })) ?? []),
    ...data.median.map((v, i) => ({
      step: i,
      median: v,
      lower: data.quantiles.q10[i],
      upper: data.quantiles.q90[i],
      type: "forecast",
    })),
  ];

  return (
    <div className="bg-white p-4 rounded-lg shadow mb-6">
      <h3 className="text-lg font-semibold mb-2">Forecast (median ± 80% CI)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#8884d8" name="History" dot={false} />
          <Line type="monotone" dataKey="median" stroke="#10b981" name="Median Forecast" />
          <Line type="monotone" dataKey="lower" stroke="#ef4444" strokeDasharray="5 5" name="10th %ile" />
          <Line type="monotone" dataKey="upper" stroke="#ef4444" strokeDasharray="5 5" name="90th %ile" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}