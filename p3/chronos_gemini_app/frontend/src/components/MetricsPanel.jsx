export default function MetricsPanel({ eval: e }) {
  return (
    <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-4 rounded-lg">
      <h4 className="font-medium text-green-800">Evaluation</h4>
      <div className="grid grid-cols-2 gap-2 text-sm mt-2">
        <div>WQL: <strong>{e.wql.toFixed(4)}</strong></div>
        <div>MASE: <strong>{e.mase.toFixed(4)}</strong></div>
        <div>vs Baseline WQL: <strong>{e.baseline_wql.toFixed(4)}</strong></div>
        <div>vs Baseline MASE: <strong>{e.baseline_mase.toFixed(4)}</strong></div>
      </div>
    </div>
  );
}