import React, { useState, useCallback } from "react";
import "./App.css";
import { SERVER_URL } from "./constants";

function App() {
  const [selectedOption, setSelectedOption] = useState("Pix2PixGan");
  const [droppedFiles, setDroppedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleSelectChange = (e) => {
    setSelectedOption(e.target.value);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files || []);
    setDroppedFiles(files);
  }, []);

  const handleFileInputChange = (e) => {
    const files = Array.from(e.target.files || []);
    setDroppedFiles(files);
  };

  const handleSubmit = async () => {
    if (droppedFiles.length === 0) return;

    const formData = new FormData();

    // Send first image or loop through all if needed
    formData.append('image', droppedFiles[0]);
    formData.append('model', selectedOption.toLowerCase().replace(/[^a-z0-9]/g, ''));

    try {
      const response = new Response(
        new ReadableStream({
          async start(controller) {
            const abortController = new AbortController();

            const fetchResponse = await fetch(SERVER_URL, {
              method: 'POST',
              body: formData,
              signal: abortController.signal,
            });

            if (!fetchResponse.ok) {
              throw new Error(`HTTP ${fetchResponse.status}`);
            }

            const reader = fetchResponse.body.getReader();

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              controller.enqueue(value);
            }
            controller.close();
          }
        })
      );

      // Create download link for restored image
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `restored_${droppedFiles[0].name}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

    } catch (error) {
      alert(error.message);
    }
  };

  return (
    <div className="app">
      <h1>Restore AI</h1>
      {false ?
        <div className="controls">
          <label>
            Choose option:
            <select value={selectedOption} onChange={handleSelectChange}>
              <option value="Pix2PixGan">Pix2Pix GAN</option>
              <option value="U-Net">U-Net</option>
              <option value="DNCNN">DNCNN</option>
              <option value="ResNet">RESNET</option>
              <option value="DCGAN">DCGAN</option>
            </select>
          </label>

          <p className="selected-info">
            Selected option: <strong>{selectedOption}</strong>
          </p>
        </div>
        : null}
      <div
        className={`drop-zone ${isDragging ? "dragging" : ""}`}
        onDragOver={handleDragOver}
        onDragEnter={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <p>Drag and drop files here</p>
        <p>or</p>
        <label className="file-input-label">
          Click to choose files
          <input
            type="file"
            multiple
            onChange={handleFileInputChange}
            style={{ display: "none" }}
          />
        </label>
      </div>

      {droppedFiles.length > 0 && (
        <div className="file-list">
          <h2>Files:</h2>
          <ul>
            {droppedFiles.map((file, index) => (
              <li key={index}>
                {file.name} ({Math.round(file.size / 1024)} KB)
              </li>
            ))}
          </ul>
        </div>
      )}

      {droppedFiles.length > 0 && (
        <div className="submitButton" onClick={handleSubmit}>
          Submit
        </div>
      )}
    </div>
  );
}

export default App;
