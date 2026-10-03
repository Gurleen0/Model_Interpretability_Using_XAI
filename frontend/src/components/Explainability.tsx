import React, { useState, useRef } from "react";

interface Explanation {
  feature: string;
  importance: number;
  type?: "positive" | "negative";
}

interface Metrics {
  faithfulness: number;
  robustness: number;
  reliability: number;
  localization: number;

}

interface AnalysisResults {
  method: string;
  totalFeatures: number;
  accuracy: number;
  processingTime: string;
  explanations: Explanation[];
  visualization?: string;
  metrics?: Metrics;
}

interface AnalysisState {
  currentStep: "upload" | "configure" | "running" | "results";
  modelFile: File | null;
  datasetFile: File | null;
  explainabilityMethod: string;
  topFeatures: number;
  generateVisualizations: boolean;
  isRunning: boolean;
  results: AnalysisResults | null;
}

const ExplainabilityDashboard: React.FC = () => {
  const [state, setState] = useState<AnalysisState>({
    currentStep: "upload",
    modelFile: null,
    datasetFile: null,
    explainabilityMethod: "SHAP (SHapley Additive exPlanations)",
    topFeatures: 10,
    generateVisualizations: true,
    isRunning: false,
    results: null,
  });

  const modelInputRef = useRef<HTMLInputElement>(null);
  const datasetInputRef = useRef<HTMLInputElement>(null);

  const updateState = (updates: Partial<AnalysisState>) => {
    setState((prev) => ({ ...prev, ...updates }));
  };

  const handleModelUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    updateState({ modelFile: file });
  };

  const handleDatasetUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    updateState({ datasetFile: file });
  };

  const runExplainabilityAnalysis = async () => {
    if (!state.modelFile || !state.datasetFile) {
      console.error("Please upload both a model and a dataset file.");
      return;
    }

    updateState({ isRunning: true, currentStep: "running" });

    try {
      const backendUrl = "http://127.0.0.1:5000";

      const modelFormData = new FormData();
      modelFormData.append("file", state.modelFile);
      const modelUploadResponse = await fetch(`${backendUrl}/upload_model`, {
        method: "POST",
        body: modelFormData,
      });
      if (!modelUploadResponse.ok) throw new Error("Model upload failed");
      const modelUploadResult = await modelUploadResponse.json();

      const datasetFormData = new FormData();
      datasetFormData.append("file", state.datasetFile);
      const datasetUploadResponse = await fetch(
        `${backendUrl}/upload_dataset`,
        {
          method: "POST",
          body: datasetFormData,
        }
      );
      if (!datasetUploadResponse.ok) throw new Error("Dataset upload failed");
      const datasetUploadResult = await datasetUploadResponse.json();

      const analysisResponse = await fetch(`${backendUrl}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          modelPath: modelUploadResult.path,
          datasetPath: datasetUploadResult.path,
          explainabilityMethod: state.explainabilityMethod,
          topFeatures: state.topFeatures,
        }),
      });

      if (!analysisResponse.ok) throw new Error("Analysis failed");

      const analysisResults = await analysisResponse.json();
      updateState({ results: analysisResults.results, currentStep: "results" });
    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Analysis failed. Please check console for details.");
    } finally {
      updateState({ isRunning: false });
    }
  };

  const canRunAnalysis = state.modelFile && state.datasetFile;

  const MetricCard = ({
    title,
    value,
    color = "purple",
  }: {
    title: string;
    value: number;
    color?: string;
  }) => (
    <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
      <div className="text-sm text-gray-400 mb-2">{title}</div>
      <div className={`text-3xl font-bold text-${color}-400`}>
        {(value * 100).toFixed(1)}%
      </div>
      <div className="mt-3 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full bg-${color}-500 transition-all duration-500`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-950 text-white font-sans">
      <header className="bg-gray-900 border-b border-gray-800 py-4">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-purple-700 rounded-xl flex items-center justify-center">
              <svg
                className="w-6 h-6 text-white"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" />
              </svg>
            </div>
            <span className="text-xl font-bold">ML Model Evaluator</span>
          </div>
          <nav className="flex space-x-2">
            <a
              href="#"
              className="px-4 py-2 bg-purple-700 text-white rounded-xl"
            >
              Explainability
            </a>
          </nav>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="mb-10">
          <h1 className="text-4xl font-bold mb-2">Explainability Analysis</h1>
          <p className="text-gray-400 text-lg">
            Analyze feature importance and understand model decision-making
            processes
          </p>
        </div>

        {state.currentStep === "results" && state.results ? (
          <div className="space-y-8">
            <div className="bg-gray-900 rounded-2xl p-8 border border-gray-800">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold">Analysis Results</h2>
                <button
                  onClick={() =>
                    updateState({ currentStep: "upload", results: null })
                  }
                  className="px-6 py-2 bg-gray-800 hover:bg-gray-700 rounded-xl transition-colors"
                >
                  New Analysis
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                <div className="bg-gray-800 rounded-xl p-5 text-center">
                  <div className="text-3xl font-bold text-purple-400">
                    {state.results.method.split(" ")[0]}
                  </div>
                  <div className="text-sm text-gray-400 mt-1">Method</div>
                </div>
                <div className="bg-gray-800 rounded-xl p-5 text-center">
                  <div className="text-3xl font-bold text-purple-400">
                    {state.results.totalFeatures}
                  </div>
                  <div className="text-sm text-gray-400 mt-1">
                    Features Analyzed
                  </div>
                </div>
                <div className="bg-gray-800 rounded-xl p-5 text-center">
                  <div className="text-3xl font-bold text-purple-400">
                    {state.results.processingTime}
                  </div>
                  <div className="text-sm text-gray-400 mt-1">
                    Processing Time
                  </div>
                </div>
              </div>

              {state.results.metrics && (
                <div className="mb-8">
                  <h3 className="text-xl font-semibold mb-4">
                    Quantitative Metrics
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <MetricCard
                      title="Faithfulness"
                      value={state.results.metrics.faithfulness}
                      color="green"
                    />
                    <MetricCard
                      title="Robustness"
                      value={state.results.metrics.robustness}
                      color="blue"
                    />
                    <MetricCard
                      title="Reliability"
                      value={state.results.metrics.reliability}
                      color="yellow"
                    />
                    <MetricCard
                      title="Localization"
                      value={state.results.metrics.localization}
                      color="red"
                    />
                  </div>
                </div>
              )}

              {state.results.visualization && (
                <div className="mb-8">
                  <h3 className="text-xl font-semibold mb-4">
                    Saliency Map Visualization
                  </h3>
                  <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
                    <img
                      src={state.results.visualization}
                      alt="Saliency Map"
                      className="w-full rounded-lg"
                    />
                  </div>
                </div>
              )}

              <div>
                <h3 className="text-xl font-semibold mb-4">
                  Feature Importance
                </h3>
                <div className="space-y-3">
                  {state.results.explanations.map((item, index) => {
                    const type =
                      item.type ??
                      (item.importance >= 0 ? "positive" : "negative");
                    const barWidth = Math.abs(item.importance) * 100;
                    return (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 bg-gray-800 rounded-lg"
                      >
                        <span className="font-medium text-gray-300 flex-1">
                          {item.feature}
                        </span>
                        <div className="flex items-center space-x-4 flex-1 justify-end">
                          <div className="w-48 bg-gray-700 rounded-full h-2.5">
                            <div
                              className={`h-2.5 rounded-full ${
                                type === "positive"
                                  ? "bg-green-500"
                                  : "bg-red-500"
                              }`}
                              style={{ width: `${barWidth}%` }}
                            />
                          </div>
                          <span className="text-sm text-gray-400 font-mono w-16 text-right">
                            {(item.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="bg-gray-900 rounded-2xl p-8 border border-gray-800">
                <h3 className="text-xl font-semibold mb-4">Upload Model</h3>
                <div className="border-2 border-dashed border-gray-700 rounded-xl p-10 text-center hover:border-purple-600 transition-colors">
                  <svg
                    className="w-16 h-16 text-gray-600 mx-auto mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  <input
                    type="file"
                    onChange={handleModelUpload}
                    className="hidden"
                    ref={modelInputRef}
                    accept=".pkl,.h5,.pt,.pth,.onnx"
                  />
                  <button
                    onClick={() => modelInputRef.current?.click()}
                    className="px-6 py-2 bg-gray-800 hover:bg-gray-700 rounded-xl transition-colors"
                  >
                    Choose File
                  </button>
                  <p className="text-xs text-gray-500 mt-3">
                    Supported: .pkl, .h5, .pt, .pth, .onnx
                  </p>
                  {state.modelFile && (
                    <div className="mt-4 p-3 bg-green-900/30 border border-green-700 rounded-lg text-sm text-green-400">
                      ✓ {state.modelFile.name}
                    </div>
                  )}
                </div>
              </div>

              <div className="bg-gray-900 rounded-2xl p-8 border border-gray-800">
                <h3 className="text-xl font-semibold mb-4">Upload Image</h3>
                <div className="border-2 border-dashed border-gray-700 rounded-xl p-10 text-center hover:border-purple-600 transition-colors">
                  <svg
                    className="w-16 h-16 text-gray-600 mx-auto mb-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  <input
                    type="file"
                    onChange={handleDatasetUpload}
                    className="hidden"
                    ref={datasetInputRef}
                    accept=".jpg,.jpeg,.png"
                  />
                  <button
                    onClick={() => datasetInputRef.current?.click()}
                    className="px-6 py-2 bg-gray-800 hover:bg-gray-700 rounded-xl transition-colors"
                  >
                    Choose File
                  </button>
                  <p className="text-xs text-gray-500 mt-3">
                    Supported: .jpg, .jpeg, .png
                  </p>
                  {state.datasetFile && (
                    <div className="mt-4 p-3 bg-green-900/30 border border-green-700 rounded-lg text-sm text-green-400">
                      ✓ {state.datasetFile.name}
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="bg-gray-900 rounded-2xl p-8 border border-gray-800">
              <h3 className="text-xl font-semibold mb-6">Configuration</h3>
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium mb-2 text-gray-300">
                    Explanation Method
                  </label>
                  <select
                    value={state.explainabilityMethod}
                    onChange={(e) =>
                      updateState({ explainabilityMethod: e.target.value })
                    }
                    className="w-full bg-gray-800 border border-gray-700 text-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-600"
                  >
                    <option value="SHAP (SHapley Additive exPlanations)">
                      SHAP (SHapley Additive exPlanations)
                    </option>
                    <option value="LIME (Local Interpretable Model-agnostic Explanations)">
                      LIME (Local Interpretable Model-agnostic Explanations)
                    </option>
                    <option value="GradCAM (Gradient-weighted Class Activation Mapping)">
                      GradCAM (Gradient-weighted Class Activation Mapping)
                    </option>
                    <option value="RISE (Randomized Input Sampling for Explanation)">
                      RISE (Randomized Input Sampling for Explanation)
                    </option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2 text-gray-300">
                    Top Features to Analyze
                  </label>
                  <input
                    type="number"
                    value={state.topFeatures}
                    onChange={(e) =>
                      updateState({
                        topFeatures: parseInt(e.target.value) || 10,
                      })
                    }
                    className="w-full bg-gray-800 border border-gray-700 text-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-600"
                    min="1"
                    max="20"
                  />
                </div>
              </div>
            </div>

            <div className="bg-gray-900 rounded-2xl p-8 border border-gray-800 text-center">
              <button
                onClick={runExplainabilityAnalysis}
                disabled={!canRunAnalysis || state.isRunning}
                className={`inline-flex items-center space-x-3 px-10 py-4 text-xl font-bold rounded-2xl transition-all ${
                  !canRunAnalysis || state.isRunning
                    ? "bg-gray-700 text-gray-500 cursor-not-allowed"
                    : "bg-purple-700 hover:bg-purple-600 text-white"
                }`}
              >
                {state.isRunning ? (
                  <>
                    <svg
                      className="animate-spin w-5 h-5"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                        className="opacity-25"
                      />
                      <path
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        className="opacity-75"
                      />
                    </svg>
                    <span>Running Analysis...</span>
                  </>
                ) : (
                  <>
                    <svg
                      className="w-6 h-6"
                      fill="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <polygon points="5,3 19,12 5,21" />
                    </svg>
                    <span>Run Explainability Analysis</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ExplainabilityDashboard;
