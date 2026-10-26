from app.users import list_users

def test_list_users_returns_rows():
    assert isinstance(list_users(), list)
