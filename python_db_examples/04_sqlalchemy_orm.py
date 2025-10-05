from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Mapped, Session, declarative_base, mapped_column
from sqlalchemy.pool import NullPool

DB = Path("example.db")
if DB.exists():
    DB.unlink()

engine = create_engine(f"sqlite:///{DB}", poolclass=NullPool, echo=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]


# create table
Base.metadata.create_all(engine)

# inserts in a Session (transaction)
with Session(engine) as s:
    s.add_all([User(name="Ada"), User(name="Grace")])
    s.commit()  # Must manually commit

# Alternative: nested begin() for auto-commit
with Session(engine) as s:
    with s.begin():  # Transaction block with auto-commit
        s.add(User(name="Jane"))
    # Auto-commits here

# read via SQL or ORM query (showing SQL here for symmetry)
with engine.connect() as conn:
    rows = conn.execute(text("SELECT id, name FROM user ORDER BY id")).mappings().all()
    print("initial:", rows)

# update inside a Session (transaction)
with Session(engine) as s:
    s.execute(
        text("UPDATE user SET name=:new WHERE name=:old"),
        {"new": "Grace Hopper", "old": "Grace"},
    )
    s.commit()

with engine.connect() as conn:
    rows = conn.execute(text("SELECT id, name FROM user ORDER BY id")).mappings().all()
    print("final:", rows)
