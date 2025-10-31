const models = [
  { id: "amazon/chronos-t5-tiny",  label: "Tiny (8M)" },
  { id: "amazon/chronos-t5-mini",  label: "Mini (20M)" },
  { id: "amazon/chronos-t5-small", label: "Small (46M)" },
  { id: "amazon/chronos-t5-base",  label: "Base (200M)" },
  { id: "amazon/chronos-t5-large", label: "Large (710M)" },
];

export default function ModelSelector({ model, setModel }) {
  return (
    <div>
      <label className="block text-sm font-medium mb-1">Chronos Model</label>
      <select
        value={model}
        onChange={(e) => setModel(e.target.value)}
        className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
      >
        {models.map((m) => (
          <option key={m.id} value={m.id}>{m.label}</option>
        ))}
      </select>
    </div>
  );
}