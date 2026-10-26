from fastapi import APIRouter
from app.db import get_conn

router = APIRouter()

@router.get("/")
def list_users():
    conn = get_conn()
    rows = conn.execute("SELECT id, email, name FROM users").fetchall()
    conn.close()
    return [dict(r) for r in rows]

@router.get("/{user_id}")
def get_user(user_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return {"error": "not found"}, 404
    return dict(row)
