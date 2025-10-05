import sqlite3
from contextlib import closing
from pathlib import Path

DB = Path("example.db")
if DB.exists():
    DB.unlink()

# IMPORTANT: The connection's __exit__ method does NOT close the connection!
# It only commits (on success) or rolls back (on exception).
# The connection stays open and must be explicitly closed or garbage collected.

with sqlite3.connect(DB) as conn:  # manages transactions (commit/rollback), NOT close!
    # Transaction 1: Create table and insert
    with conn:  # manages transaction lifecycle (commit/rollback)
        with closing(conn.cursor()) as cur:
            cur.execute("CREATE TABLE user(id INTEGER PRIMARY KEY, name TEXT)")
            cur.execute("INSERT INTO user(name) VALUES (?)", ("Ada",))
            cur.execute("INSERT INTO user(name) VALUES (?)", ("Grace",))
    # Transaction 1 commits here, connection stays open

    # Read data (connection still open)
    with closing(conn.cursor()) as cur:
        cur.execute("SELECT id, name FROM user ORDER BY id")
        print("initial:", cur.fetchall())

    # Transaction 2: Update
    try:
        with conn:
            with closing(conn.cursor()) as cur:
                cur.execute(
                    "UPDATE user SET name=? WHERE name=?", ("Grace Hopper", "Grace")
                )
    except Exception:
        # with-conn would auto-rollback, this is just to show where you'd react
        raise
    # Transaction 2 commits here, connection stays open

    # Read final state
    with closing(conn.cursor()) as cur:
        cur.execute("SELECT id, name FROM user ORDER BY id")
        print("final:", cur.fetchall())
# Connection still open after exiting 'with' block!

print("*********** Connection still works after 'with' block ***********")
print(conn.execute("SELECT * FROM user").fetchall())
conn.close()  # Must explicitly close
