"""Integrity checks on the static drug registry — these are cheap to run
and catch a whole class of "typo in a hardcoded list" bugs (duplicate
names, a class combination referencing a class with zero drugs, etc.)
that would otherwise only surface much later during ingestion or
inference."""

from app.core.drug_registry import DISCOURAGED_COMBINATIONS, HYPERTENSION_DRUGS, STANDARD_COMBINATIONS
from app.schemas.compound import DrugClass


def test_no_duplicate_drug_names():
    names = [d["name"] for d in HYPERTENSION_DRUGS]
    assert len(names) == len(set(names)), f"duplicate names found: {[n for n in names if names.count(n) > 1]}"


def test_all_drug_classes_are_valid_enum_members():
    for drug in HYPERTENSION_DRUGS:
        assert isinstance(drug["drug_class"], DrugClass)
        assert drug["drug_class"] != DrugClass.UNKNOWN


def test_gene_symbols_present_and_nonempty():
    for drug in HYPERTENSION_DRUGS:
        assert drug["gene_symbol"], f"{drug['name']} has an empty gene_symbol"


def test_each_class_has_at_least_one_drug():
    classes_present = {d["drug_class"] for d in HYPERTENSION_DRUGS}
    for cls in DrugClass:
        if cls == DrugClass.UNKNOWN:
            continue
        assert cls in classes_present, f"{cls} has no drugs in HYPERTENSION_DRUGS"


def test_each_class_maps_to_exactly_one_gene_symbol():
    """A class shouldn't accidentally end up split across two different
    target genes — that would silently break the gene->target->potency
    resolution pipeline in build_registry.py."""
    class_to_genes: dict[DrugClass, set[str]] = {}
    for drug in HYPERTENSION_DRUGS:
        class_to_genes.setdefault(drug["drug_class"], set()).add(drug["gene_symbol"])
    for cls, genes in class_to_genes.items():
        assert len(genes) == 1, f"{cls} maps to multiple gene symbols: {genes}"


def test_standard_and_discouraged_combinations_reference_real_classes():
    classes_present = {d["drug_class"] for d in HYPERTENSION_DRUGS}
    for pair in STANDARD_COMBINATIONS + DISCOURAGED_COMBINATIONS:
        assert len(pair) == 2
        for cls in pair:
            assert cls in classes_present, f"{cls} in a combination list but has no drugs in the registry"


def test_standard_and_discouraged_combinations_do_not_overlap():
    """A class pair shouldn't be simultaneously endorsed and flagged as
    dangerous — that would be a direct contradiction in the report logic."""
    standard_pairs = {frozenset(p) for p in STANDARD_COMBINATIONS}
    discouraged_pairs = {frozenset(p) for p in DISCOURAGED_COMBINATIONS}
    assert standard_pairs.isdisjoint(discouraged_pairs)


def test_ace_arb_dual_raas_is_flagged_discouraged():
    """Pin the one safety-critical fact this registry currently encodes —
    if this ever gets refactored away accidentally, this test should fail
    loudly rather than the report silently stop warning about it."""
    assert frozenset({DrugClass.ACE_INHIBITOR, DrugClass.ARB}) in {frozenset(p) for p in DISCOURAGED_COMBINATIONS}
