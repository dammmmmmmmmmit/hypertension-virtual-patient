"""Bootstrap the ingestion-cache schema. Run: uv run python -m app.db.init_db
See models.py docstring re: no Alembic chain yet — this is create_all(),
fine for a schema this small and still-moving, not a substitute for real
migrations once it stabilizes."""

import asyncio

from app.db.models import Base
from app.db.session import engine


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    asyncio.run(init_db())
    print("resolved_compounds table ready")
