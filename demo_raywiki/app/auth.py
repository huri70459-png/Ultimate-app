import jwt
from fastapi import APIRouter

router = APIRouter()
SECRET = "changeme"

@router.post("/login")
def login(username: str, password: str):
    token = jwt.encode({"sub": username}, SECRET, algorithm="HS256")
    return {"access_token": token}

def verify_token(token: str) -> dict:
    return jwt.decode(token, SECRET, algorithms=["HS256"])
