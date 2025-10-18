# 🧠 AI Cost Estimator — CLI & Dockerized Pipeline

## 🚀 Overview

AI Cost Estimator is an interactive, terminal-based pipeline that collects app requirements, detects relevant tasks from a catalog, and generates minimum/maximum plan-and-cost estimates.

Workflow:
````
intake (interactive) → task_detect → plan_and_cost
````

Each run creates a unique folder under data/out/<slug>-YYYYMMDD_HHMMSS containing all output artifacts:

* intake_*.json
* detected_*.json
* cost_report_*.{json,csv,md}

## 📦 Repository Highlights

| File/Folder               | Description                                                                                                |
| ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `docker-compose.yml`      | Defines the containerized app and mounts `data/` for persistent outputs.                                   |
| `Dockerfile`              | Builds a lightweight Python 3.12 image and sets the default entrypoint (`python -m src.app.cli.pipeline`). |
| `src/app/cli/pipeline.py` | Orchestrates the full end-to-end flow.                                                                     |
| `data/`                   | Contains `raw/` (catalog inputs) and `out/` (generated reports).                                           |
| `docs/ARCHITECTURE.md`    | High-level architecture explanation.                                                                       |

Note: The project includes config.py and config.toml for future extension,
but they are optional — the pipeline runs fine without them.

## 🧰 Prerequisites

Docker
 (includes Docker Compose on Windows/macOS)

Internet connection (for building dependencies)

## 🪄 Quick Start — Run with Docker Compose

1️⃣ Clone the repository
````
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
````

2️⃣ Build the Docker image
````
docker compose build
````

3️⃣ Run interactively inside Docker
````
docker compose run --rm app
````
You’ll be prompted for inputs in your terminal.
The app will automatically run intake → detection → cost estimation
and save outputs to your host at:
````
data/out/<name>-<timestamp>/
````

| Location         | Path                |
| ---------------- | ------------------- |
| On host          | `./data/out/...`    |
| Inside container | `/app/data/out/...` |

## ⚙️ Alternative: One-Line Run (auto-build + interactive)
````
docker compose up --build
````

This will build the image (if needed) and start the pipeline automatically.
If your terminal doesn’t accept input in this mode, use docker compose run --rm app instead.

## 🧩 Run Individual Steps

Run only specific stages inside Docker (optional):

| Step        | Command                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------- |
| Intake      | `docker compose run --rm app python -m src.app.core.cli_intake`                                         |
| Detect      | `docker compose run --rm app python -m src.app.core.task_detect --run-dir "/app/data/out/<your-run>"`   |
| Plan & Cost | `docker compose run --rm app python -m src.app.core.plan_and_cost --run-dir "/app/data/out/<your-run>"` |

## 💻 Local (non-Docker) Development

If you prefer running directly in Python:

1. Create virtual environment
````
python -m venv .venv
.\.venv\Scripts\activate    # Windows
source .venv/bin/activate   # macOS/Linux
````
2. Install dependencies
````
pip install --upgrade pip
pip install -r requirements
````
3. Run pipeline
````
python -m src.app.cli.pipeline
````
Outputs will still be written to data/out/.

### 🗂 Data and Volume Mounting

* The Compose file mounts your local ./data directory to the container’s /app/data.

* This ensures all results generated in Docker persist on your machine.
````
Host:        ./data/out/...
Container:   /app/data/out/...
````

### ⚙️ Configuration (Optional)

* src/app/config.py automatically loads defaults.

* If config.toml exists, it can override parameters like OUTPUT_DIR.

* You do not need to edit any config file for normal use.

### 🧩 Troubleshooting

| Issue                              | Fix                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| **No input prompt appears**        | Use `docker compose run --rm app` instead of `docker compose up`.               |
| **Permission errors on new files** | Adjust permissions for the `data/` folder (`chmod -R 777 data` on Linux/macOS). |
| **Dependencies updated**           | Rebuild image: `docker compose build`.                                          |

## 🏗️ Project Status

This is an interactive CLI-first prototype for the Hyperskill Hackathon (October 2025).
Future versions may integrate:

* A FastAPI web frontend

* A structured config.toml workflow

* Extended cost-model support

## 📘 Documentation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for high-level design and pipeline flow.

## 🪪 License

See the [LICENSE](LICENSE) file for details.
