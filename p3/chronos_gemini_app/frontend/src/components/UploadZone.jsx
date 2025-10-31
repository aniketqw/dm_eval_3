import { Upload } from "lucide-react";

export default function UploadZone({ setDataset, sessionId }) {
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    const form = new FormData();
    form.append("file", file);
    form.append("session_id", sessionId);

    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    setDataset({ fileName: file.name, analysis: data.analysis });
  };

  return (
    <div className="border-2 border-dashed border-indigo-300 rounded-xl p-6 text-center">
      <Upload className="mx-auto w-12 h-12 text-indigo-500 mb-2" />
      <input type="file" accept=".csv,.parquet" onChange={handleUpload} className="hidden" id="upload" />
      <label htmlFor="upload" className="cursor-pointer text-indigo-600 font-medium">
        Click to upload CSV / Parquet
      </label>
    </div>
  );
}