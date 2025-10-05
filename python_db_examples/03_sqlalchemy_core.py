from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

DB = Path("example.db")
if DB.exists():
    DB.unlink()

# Engine with NullPool = great for scripts (fresh connection per use)
engine = create_engine(f"sqlite:///{DB}", poolclass=NullPool, echo=False)

# SQLAlchemy guarantees connection closure: both engine.connect() and engine.begin()
# close connections automatically on __exit__ (unlike raw sqlite3 where with conn: only commits/rollbacks)

# schema + inserts in one transactional block
with engine.begin() as conn:  # begin/commit automatically
    conn.execute(text("CREATE TABLE user(id INTEGER PRIMARY KEY, name TEXT)"))
    conn.execute(text("INSERT INTO user(name) VALUES (:n)"), {"n": "Ada"})
    conn.execute(text("INSERT INTO user(name) VALUES (:n)"), {"n": "Grace"})


# read (no explicit transaction needed for SELECT)
with engine.connect() as conn:
    # Without .mappings() - returns tuples
    result = conn.execute(text("SELECT id, name FROM user ORDER BY id"))
    rows_tuples = result.all()
    print("Tuples:", rows_tuples)
    print("Access by index:", rows_tuples[0][1])  # 'Ada'

    # With .mappings() - returns dict-like objects
    result = conn.execute(text("SELECT id, name FROM user ORDER BY id"))
    rows_dicts = result.mappings().all()
    print("\nMappings:", rows_dicts)
    print("Access by name:", rows_dicts[0]["name"])  # 'Ada'

    # Different fetch methods
    print("\n--- Fetch methods ---")

    # .all() - fetch all rows
    all_rows = conn.execute(text("SELECT name FROM user")).mappings().all()
    print("all():", all_rows)

    # .first() - fetch first row or None
    first_row = (
        conn.execute(text("SELECT name FROM user ORDER BY id")).mappings().first()
    )
    print("first():", first_row)

    # .one() - fetch exactly one row (error if 0 or 2+)
    # Use for assertions/validation when you expect exactly 1 result
    # first() returns first row or None (doesn't care about total count)
    one_row = conn.execute(text("SELECT name FROM user WHERE id = 1")).mappings().one()
    print("one():", one_row)

    # .scalars() - extracts only first column values (no dict/row wrapping)
    # If multiple columns, scalars() ignores all but first column
    names = conn.execute(text("SELECT name FROM user ORDER BY id")).scalars().all()
    print("scalars().all():", names)  # ['Ada', 'Grace'] - clean list, not dicts

# update in a small transaction
with engine.begin() as conn:
    conn.execute(
        text("UPDATE user SET name=:new WHERE name=:old"),
        {"new": "Grace Hopper", "old": "Grace"},
    )

with engine.connect() as conn:
    rows = conn.execute(text("SELECT id, name FROM user ORDER BY id")).mappings().all()
    print("final:", rows)
