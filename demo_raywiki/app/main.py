from fastapi import FastAPI
from app.users import router as users_router
from app.auth import router as auth_router

app = FastAPI(title="RayWiki")
app.include_router(users_router, prefix="/users")
app.include_router(auth_router, prefix="/auth")

@app.get("/health")
def health():
    return {"status": "ok"}
