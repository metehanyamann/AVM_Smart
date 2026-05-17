from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        app_dir=str(PROJECT_ROOT),
    )
