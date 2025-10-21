# 🧠 AI Cost Estimator — CLI & Dockerized Pipeline

## 🚀 Overview

AI Cost Estimator is an interactive, terminal-based pipeline that collects app requirements, detects relevant tasks from a catalog (using LLMs), and generates minimum/maximum cost estimates.

**Workflow:**
```
intake (interactive) → task_detect (LLM-based) → plan_and_cost
```

Each run automatically creates a timestamped folder under:
```
data/out/<project_name>-YYYYMMDD_HHMMSS/
```

This folder stores all generated artifacts:
- `intake_*.json` — user input
- `selected_tasks_*.json` — detected tasks
- `explain_*.md` — reasoning summary
- `cost_breakdown_*.json` — per-task cost details
- `cost_summary_*.md` — formatted final report
- `run_meta.json` — metadata (provider, timestamp, etc.)

---
## 🎥 Demo Video

You can watch the complete 30-minute demo video on Google Drive using the link below:  
👉 **[View the full demo on Google Drive](https://drive.google.com/file/d/1R5a-CEUYsN_agAPOhq_4JV1Nux45H4XW/view?usp=sharing)**

> 💡 Tip: For a smoother viewing experience, watch the video at **2× speed**.
---
## 📂 Repository Structure
```
.
├── data
│   ├── raw                     # Input catalogs
│   │   └── AI_Cost_Estimator_Catalog_UPDATED.xlsx
│   └── out                     # Generated outputs (auto-created)
├── docs
│   └── ARCHITECTURE.md         # System design and workflow
├── src
│   └── app
│       ├── cli/                # CLI entrypoints
│       ├── core/               # Core logic (intake, detect, cost)
│       ├── llm/                # LLM provider configs and selectors
│       └── utils/              # Internal helpers
├── utils
│   ├── example_prompts.json    # Example prompts to test the app
│   ├── generate_root_folder_tree.py
│   └── clean_folder_structure.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements
├── LICENSE
└── README.md
```

---

## 🧰 Prerequisites

- **Docker** (includes Docker Compose on Windows/macOS)
- **Internet connection** (required for LLM task detection)

---

## ⚙️ Environment Setup

Copy the example environment file and fill your API keys:
```bash
cp .env.example .env
```

`.env.example` contains:
```dotenv
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
PERPLEXITY_API_KEY=your-perplexity-api-key
GROK_API_KEY=your-grok-api-key
```

> **Note:**  
> When using an LLM for task detection, slight variability in results is expected — even with fixed parameters (temperature=0, top_p=1).  
> This is more noticeable with **Perplexity**, while **OpenAI** tends to be more stable.  
>  
> The **Google** and **Grok** integrations are experimental and not fully tested yet — you can try them and share feedback.

---

## 🪄 Quick Start — Run with Docker Compose

1️⃣ **Build the image**
```bash
docker compose build
```

2️⃣ **Run interactively**
```bash
docker compose run --rm app
```

You’ll be prompted for project details directly in your terminal.  
The pipeline will automatically run all stages and write outputs to:
```
data/out/<name>-<timestamp>/
```

| Location         | Path                |
| ---------------- | ------------------- |
| On host          | `./data/out/...`    |
| Inside container | `/app/data/out/...` |

---

## 💡 Example Prompts

You can explore **sample prompts** for various application domains in:
```
utils/example_prompts.json
```

Use any of them to test the MVP and see how the **task detection** adapts across different use cases.

---

## 🧩 Optional — Run Individual Steps

| Step        | Command                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------- |
| Intake       | `docker compose run --rm app python -m src.app.core.cli_intake`                                         |
| Detect Tasks | `docker compose run --rm app python -m src.app.core.task_detect --run-dir "/app/data/out/<your-run>"`   |
| Plan & Cost  | `docker compose run --rm app python -m src.app.core.plan_and_cost --run-dir "/app/data/out/<your-run>"` |

---

## 💻 Local Development (without Docker)

If you prefer running locally:

```bash
python -m venv .venv
.\.venv\Scripts\activate       # Windows
source .venv/bin/activate        # macOS/Linux

pip install --upgrade pip
pip install -r requirements

python -m src.app.cli.pipeline
```

Outputs will still be saved to `data/out/`.

---
## 📘 Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for high-level design and pipeline flow.

For a detailed overview of current issues, known limitations, and possible improvements,  
please see [docs/CURRENT_ISSUES_AND_POSSIBLE_SOLUTIONS.md](docs/CURRENT_ISSUES_AND_POSSIBLE_SOLUTIONS.md).

---

## 🪪 License

See the [LICENSE](LICENSE) file for license details.

---

## 🏗️ Project Info

This CLI-based prototype was developed for the **Hyperskill Hackathon — October 2025**.  
Future updates will include:
- Automated LLM-based report generation
- Expanded task catalog
- Web dashboard for visualization
