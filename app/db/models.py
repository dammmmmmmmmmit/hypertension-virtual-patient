"""
SQLAlchemy ORM models for the ingestion cache. One table so far:
`resolved_compounds` — one row per registry drug, holding everything the
ingestion clients (ChEMBL/PubChem/SIDER/RDKit) resolved for it, so a
training-dataset rebuild or an agent `resolve_entities` lookup doesn't
re-hit external APIs every time.

Migration note: no Alembic migration chain is set up yet (deliberate scope
cut — see DECISIONS.md). Schema is small and still moving during Week 1
dataset assembly; `init_db.py`'s `create_all()` is enough for now. Add
Alembic before this schema is considered stable / before a second
developer touches it.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ResolvedCompound(Base):
    __tablename__ = "resolved_compounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    drug_class: Mapped[str] = mapped_column(String)
    gene_symbol: Mapped[str] = mapped_column(String)

    # ChEMBL
    chembl_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    parent_chembl_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    target_chembl_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    canonical_smiles: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Potency — nullable by design (thiazides), never fabricated. See
    # potency_utils.py and DECISIONS.md #1.
    mean_potency: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    n_valid_potency_records: Mapped[int] = mapped_column(Integer, default=0)

    # PubChem / SIDER linkage
    pubchem_cid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    side_effects: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # RDKit descriptors, stored as a JSON blob rather than one column per
    # descriptor — keeps this table stable if the descriptor set grows.
    rdkit_descriptors: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    resolved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
