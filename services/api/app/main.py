from fastapi import FastAPI

app = FastAPI(
    title="YatraLink API",
    description="Backend API for YatraLink railway journey optimization.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "yatrallink-api",
        "version": "0.1.0",
    }
