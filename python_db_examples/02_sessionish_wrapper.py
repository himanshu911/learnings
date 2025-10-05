import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB = Path("example.db")
if DB.exists():
    DB.unlink()


@contextmanager
def session(db_path: Path):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN")  # explicit BEGIN for clarity
        yield conn  # caller runs multiple statements
        conn.commit()  # success -> commit
    except Exception:
        conn.rollback()  # error -> rollback
        raise
    finally:
        conn.close()


# use it
with session(DB) as s:
    s.execute("CREATE TABLE user(id INTEGER PRIMARY KEY, name TEXT)")
    s.execute("INSERT INTO user(name) VALUES (?)", ("Ada",))
    s.execute("INSERT INTO user(name) VALUES (?)", ("Grace",))

# read with a quick, plain connection
with sqlite3.connect(DB) as conn:
    print("initial:", conn.execute("SELECT id, name FROM user ORDER BY id").fetchall())
conn.close()  # Must explicitly close

# update within a transaction (via session)
with session(DB) as s:
    s.execute("UPDATE user SET name=? WHERE name=?", ("Grace Hopper", "Grace"))

with sqlite3.connect(DB) as conn:
    print("final:", conn.execute("SELECT id, name FROM user ORDER BY id").fetchall())
conn.close()  # Must explicitly close
