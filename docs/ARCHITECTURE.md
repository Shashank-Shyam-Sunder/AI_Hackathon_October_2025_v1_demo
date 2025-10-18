The detailed architecture explanation and related artifacts (for example, diagrams) should be here.

# 🗂️ Project Folder Structure

This document provides an overview of the repository structure and describes the purpose of each major folder and file.

```
.
├── .env.example
├── .gitignore
├── Dockerfile
├── clean_folder_structure.txt
├── config.toml
├── data
│   ├── out
│   └── raw
│       ├── AI_Cost_Estimator_Task_Catalog_200rows_v2.csv
│       └── AI_Cost_Estimator_Task_Catalog_200rows_v2.xlsx
├── docker-compose.yml
├── docs
│   └── ARCHITECTURE.md
├── generate_root_folder_tree.py
├── requirements
├── src
│   └── app
│       ├── __init__.py
│       ├── cli
│       │   ├── __init__.py
│       │   ├── commands.py
│       │   └── pipeline.py
│       ├── config.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── cli_intake.py
│       │   ├── plan_and_cost.py
│       │   └── task_detect.py
│       └── utils
│           ├── __init__.py
│           └── io_paths.py
└── tools
```

---

## 📁 Root Directory

| File / Folder | Description |
|----------------|--------------|
| `.env.example` | Template for environment variables. Copy this as `.env` and fill in your configuration. |
| `.gitignore` | Specifies files and directories that should not be tracked by Git. |
| `Dockerfile` | Defines the image configuration used to containerize the application. |
| `docker-compose.yml` | Defines multi-container Docker setup (services, networks, volumes). |
| `config.toml` | Central configuration file for app-wide parameters. |
| `clean_folder_structure.txt` | Text export of the project’s cleaned directory structure. |
| `generate_root_folder_tree.py` | Script used to generate folder tree documentation automatically. |
| `requirements` | Directory or file listing Python dependencies (e.g., `requirements.txt` or submodules). |
| `tools/` | Utility scripts or helper functions used for automation or maintenance. |

---

## 📊 Data Folder

| Path | Description |
|------|--------------|
| `data/raw/` | Contains raw input datasets for the project. |
| `data/out/` | Contains generated or processed output files (predictions, results, reports). |
| `AI_Cost_Estimator_Task_Catalog_200rows_v2.csv` | Raw dataset of task catalog used for cost estimation. |
| `AI_Cost_Estimator_Task_Catalog_200rows_v2.xlsx` | Same dataset as Excel version for manual inspection. |

---

## 📘 Docs Folder

| File | Description |
|-------|--------------|
| `docs/ARCHITECTURE.md` | Detailed system architecture documentation including diagrams and component descriptions. |

---

## 🧠 Source Code (`src/app/`)

### 🔹 CLI Layer

| File | Description |
|------|--------------|
| `src/app/cli/commands.py` | Defines available CLI commands and their arguments. |
| `src/app/cli/pipeline.py` | Main entry point for running the cost estimation pipeline from CLI. |

### 🔹 Core Logic

| File | Description |
|------|--------------|
| `src/app/core/cli_intake.py` | Parses and validates CLI inputs before computation. |
| `src/app/core/plan_and_cost.py` | Main cost computation logic combining plan generation and cost formulas. |
| `src/app/core/task_detect.py` | Detects relevant tasks based on user input or data. |

### 🔹 Utilities

| File | Description |
|------|--------------|
| `src/app/utils/io_paths.py` | Defines consistent input/output path references across the codebase. |
| `src/app/config.py` | Loads and manages application-level configuration (environment, paths, etc.). |

---

## 🧩 Other Folders

| Folder | Description |
|---------|--------------|
| `tools/` | Reserved for future utility or deployment-related scripts (e.g., CI/CD helpers, test scripts). |

---

## 🏗️ Developer Notes

- Run the project locally with:
  ```bash
  docker-compose up
  ```
- To rebuild after code changes:
  ```bash
  docker-compose up --build
  ```
- For CLI usage:
  ```bash
  python -m src.app.cli.pipeline
  ```

---

## 📄 Version Control & Config

- **Environment variables** → managed via `.env`  
- **Dependencies** → managed via `requirements/`  
- **Containerization** → via `Dockerfile` and `docker-compose.yml`  

---

### ✅ Recommendation
Keep this document updated whenever new folders or modules are added — ideally regenerate it automatically using `generate_root_folder_tree.py`.

