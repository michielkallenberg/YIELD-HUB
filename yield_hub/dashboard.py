from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .artifacts import LINEAR_MODELS, SUPPORTED_MODELS, TRANSFORMER_MODELS
from .predictor import predict
from .settings import REPO_ROOT


class PredictionRequest(BaseModel):
    model_type: str
    country: str
    crop: str
    checkpoint_name: Optional[str] = None
    cybench_root: Optional[str] = None
    output_dir: Optional[str] = None


def create_app() -> FastAPI:
    app = FastAPI(title="YIELD-HUB Dashboard", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        model_options = "\n".join(
            f'<option value="{model}">{model}</option>' for model in SUPPORTED_MODELS
        )
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YIELD-HUB Dashboard</title>
  <style>
    :root {{
      --bg: #f5f0e8;
      --panel: #fffaf3;
      --ink: #122117;
      --muted: #5e6d61;
      --line: #d9ceb8;
      --accent: #286c4d;
      --accent-2: #d98f3d;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217, 143, 61, 0.14), transparent 30%),
        radial-gradient(circle at top right, rgba(40, 108, 77, 0.14), transparent 28%),
        linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
    }}
    .shell {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 40px 20px 60px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      margin-bottom: 28px;
    }}
    .eyebrow {{
      display: inline-block;
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(40, 108, 77, 0.09);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 5vw, 3.6rem);
      line-height: 0.95;
      font-family: "IBM Plex Serif", Georgia, serif;
      max-width: 10ch;
    }}
    .lead {{
      margin: 0;
      max-width: 56rem;
      color: var(--muted);
      font-size: 1.05rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 20px;
    }}
    .card {{
      background: rgba(255, 250, 243, 0.88);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px;
      box-shadow: 0 20px 40px rgba(18, 33, 23, 0.06);
      backdrop-filter: blur(8px);
    }}
    .card h2 {{
      margin: 0 0 14px;
      font-size: 1.15rem;
    }}
    .row {{
      display: grid;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .row.double {{
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    label {{
      font-size: 0.9rem;
      color: var(--muted);
    }}
    input, select {{
      width: 100%;
      box-sizing: border-box;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      color: var(--ink);
      font-size: 0.96rem;
    }}
    button {{
      width: 100%;
      padding: 13px 16px;
      border: none;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--accent) 0%, #21593f 100%);
      color: white;
      font-size: 0.98rem;
      font-weight: 600;
      cursor: pointer;
    }}
    button:hover {{
      filter: brightness(1.05);
    }}
    .hint {{
      margin-top: 12px;
      font-size: 0.85rem;
      color: var(--muted);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.7);
    }}
    .metric .label {{
      display: block;
      color: var(--muted);
      font-size: 0.8rem;
      margin-bottom: 8px;
    }}
    .metric .value {{
      font-size: 1.2rem;
      font-weight: 700;
    }}
    pre {{
      overflow: auto;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #152118;
      color: #eff7ee;
      font-size: 0.86rem;
      min-height: 280px;
      margin: 0;
    }}
    @media (max-width: 860px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .metrics {{
        grid-template-columns: 1fr;
      }}
      .row.double {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <span class="eyebrow">YIELD-HUB</span>
      <h1>Forecast crop yield from private Hub checkpoints.</h1>
      <p class="lead">
        Use the package-backed prediction pipeline to pull a trained checkpoint from Hugging Face,
        load local CY-BENCH-style data, and inspect the output without dropping back to the terminal.
      </p>
    </section>
    <div class="grid">
      <section class="card">
        <h2>Run Prediction</h2>
        <form id="predict-form">
          <div class="row">
            <label for="model_type">Model</label>
            <select id="model_type" name="model_type">{model_options}</select>
          </div>
          <div class="row double">
            <div>
              <label for="country">Country</label>
              <input id="country" name="country" value="NL" />
            </div>
            <div>
              <label for="crop">Crop</label>
              <input id="crop" name="crop" value="maize" />
            </div>
          </div>
          <div class="row">
            <label for="checkpoint_name">Checkpoint Name (optional)</label>
            <input id="checkpoint_name" name="checkpoint_name" placeholder="Auto-resolve from config-and-results.csv" />
          </div>
          <div class="row">
            <label for="cybench_root">CYBENCH Root (optional)</label>
            <input id="cybench_root" name="cybench_root" placeholder="Uses YIELD-HUB/data first, then env/fallbacks" />
          </div>
          <div class="row">
            <label for="output_dir">Output Directory</label>
            <input id="output_dir" name="output_dir" value="{REPO_ROOT / 'wrappers' / 'data'}" />
          </div>
          <button type="submit">Generate Predictions</button>
          <p class="hint">
            Linear models: {", ".join(sorted(LINEAR_MODELS))}<br>
            Transformer models: {", ".join(sorted(TRANSFORMER_MODELS))}
          </p>
        </form>
      </section>
      <section class="card">
        <h2>Latest Result</h2>
        <div class="metrics">
          <div class="metric"><span class="label">Rows</span><span class="value" id="rows">-</span></div>
          <div class="metric"><span class="label">Mean Absolute Error</span><span class="value" id="mae">-</span></div>
          <div class="metric"><span class="label">Output</span><span class="value" id="output">-</span></div>
        </div>
        <pre id="result-box">Submit a prediction job to see the preview here.</pre>
      </section>
    </div>
  </div>
  <script>
    const form = document.getElementById("predict-form");
    const resultBox = document.getElementById("result-box");
    const rowsEl = document.getElementById("rows");
    const maeEl = document.getElementById("mae");
    const outputEl = document.getElementById("output");

    form.addEventListener("submit", async (event) => {{
      event.preventDefault();
      resultBox.textContent = "Running prediction...";
      rowsEl.textContent = "-";
      maeEl.textContent = "-";
      outputEl.textContent = "-";

      const payload = Object.fromEntries(new FormData(form).entries());
      for (const key of ["checkpoint_name", "cybench_root"]) {{
        if (!payload[key]) delete payload[key];
      }}

      try {{
        const response = await fetch("/predict", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify(payload),
        }});
        const data = await response.json();
        if (!response.ok) {{
          throw new Error(data.detail || "Prediction failed");
        }}
        rowsEl.textContent = data.row_count;
        maeEl.textContent = data.mean_abs_error.toFixed(4);
        outputEl.textContent = data.output_path;
        resultBox.textContent = JSON.stringify(data.preview, null, 2);
      }} catch (error) {{
        resultBox.textContent = error.message;
      }}
    }});
  </script>
</body>
</html>"""

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/models")
    def list_models() -> dict:
        return {
            "supported_models": SUPPORTED_MODELS,
            "transformer_models": sorted(TRANSFORMER_MODELS),
            "linear_models": sorted(LINEAR_MODELS),
        }

    @app.post("/predict")
    def run_prediction(request: PredictionRequest) -> dict:
        output_dir = Path(request.output_dir) if request.output_dir else REPO_ROOT / "wrappers" / "data"
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_df = predict(
            model_type=request.model_type,
            country=request.country,
            crop=request.crop,
            checkpoint_name=request.checkpoint_name,
            cybench_root=request.cybench_root,
        )

        output_path = output_dir / f"{request.model_type}_{request.crop}_{request.country}_predictions.csv"
        predictions_df.to_csv(output_path, index=False)

        return {
            "output_path": str(output_path),
            "row_count": int(len(predictions_df)),
            "mean_abs_error": float(predictions_df["abs_error"].mean()),
            "preview": predictions_df.head(15).to_dict(orient="records"),
        }

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("yield_hub.dashboard:app", host="127.0.0.1", port=8000, reload=False)
