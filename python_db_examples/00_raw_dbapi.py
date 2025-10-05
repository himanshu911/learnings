import sqlite3
from pathlib import Path

DB = Path("example.db")
if DB.exists():
    DB.unlink()


def check_db(msg: str) -> None:
    c = sqlite3.connect(DB)
    print(f"{msg}: {c.execute('SELECT * FROM user').fetchall()}")
    c.close()


conn = sqlite3.connect(DB)
cur = None
try:
    cur = conn.cursor()
    cur.execute("CREATE TABLE user(id INTEGER PRIMARY KEY, name TEXT)")
    cur.executemany(
        "INSERT INTO user(name) VALUES (?)",
        [("Ada",), ("Grace",), ("Alan",), ("Margaret",)],
    )
    check_db("Before commit")
    conn.commit()
    check_db("After commit")

    # Cursor-specific features
    print(f"\nRows affected: {cur.rowcount}")
    print(f"Column info: {cur.description}\n")

    # Different fetch methods
    cur.execute("SELECT * FROM user ORDER BY id")
    print(f"fetchone(): {cur.fetchone()}")
    print(f"fetchmany(2): {cur.fetchmany(2)}")
    print(f"fetchall(): {cur.fetchall()}\n")

    # Multiple cursors on same connection (useful for nested queries)
    cur.execute("SELECT * FROM user WHERE id = 1")
    cur2 = conn.cursor()
    cur2.execute("SELECT * FROM user WHERE id = 2")
    print(f"Cursor 1: {cur.fetchone()}")
    print(f"Cursor 2: {cur2.fetchone()}\n")
    cur2.close()

    # Transaction
    try:
        cur.execute("UPDATE user SET name=? WHERE name=?", ("Grace Hopper", "Grace"))
        print(f"UPDATE affected {cur.rowcount} row(s)")
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    check_db("Final")
finally:
    if cur:
        cur.close()
    conn.close()
