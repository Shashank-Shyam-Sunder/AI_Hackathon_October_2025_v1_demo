from __future__ import annotations
import os, sys
from pydantic import BaseModel

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

CONFIG_TOML = os.getenv("PRICING_CONFIG", "config.toml")

class AppSettings(BaseModel):
    env: str = "dev"
    log_level: str = "INFO"
    output_dir: str = "data/out"

class PricingSettings(BaseModel):
    # Put defaults your code uses here as fields later if you want to validate them.
    pass

class Settings(BaseModel):
    app: AppSettings = AppSettings()
    pricing: PricingSettings = PricingSettings()

def load_settings() -> Settings:
    data = {}
    try:
        with open(CONFIG_TOML, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        pass

    # env overrides
    app = data.get("app", {})
    app["env"] = os.getenv("ENV", app.get("env", "dev"))
    app["log_level"] = os.getenv("LOG_LEVEL", app.get("log_level", "INFO"))
    app["output_dir"] = os.getenv("OUTPUT_DIR", app.get("output_dir", "data/out"))
    data["app"] = app

    return Settings(**data)

settings = load_settings()
