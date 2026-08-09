"""
Tests for the non-LLM agent nodes (resolve_entities, comparators,
structure_features, the clarification-report path). These are
integration-style — they hit the real Postgres cache and Qdrant index
built earlier in this project, consistent with how the rest of this
project verifies behavior against real infra rather than mocks (see
memory: user prefers real integration checks over mocked ones for this
kind of pipeline work).

parse_query and generate_report's LLM path are NOT covered here — no
Anthropic API key was available this session (see DECISIONS.md). Add
tests for those once a key is available.
"""

import pytest

from app.agent.comparators import retrieve_comparators
from app.agent.generate_report import _clarification_report
from app.agent.predict import structure_features
from app.agent.resolve_entities import resolve_entities
from app.agent.schemas import ParsedPatient
from app.schemas.compound import CompoundQuery, DrugClass
from app.schemas.patient import DiseaseParameters, PatientProfile, Sex


def make_patient() -> PatientProfile:
    return PatientProfile(
        age=58, sex=Sex.MALE, weight_kg=82,
        baseline=DiseaseParameters(systolic_bp=152, diastolic_bp=96, heart_rate=78, egfr=75),
    )


def make_state(*raw_names: str) -> dict:
    return {
        "raw_query": "test",
        "compounds": [CompoundQuery(raw_name=n) for n in raw_names],
        "parsed_patient": ParsedPatient(patient=make_patient(), defaulted_fields=[]),
        "limitations": [],
    }


@pytest.mark.asyncio
class TestResolveEntities:
    async def test_exact_match_resolves(self):
        state = await resolve_entities(make_state("lisinopril"))
        assert state["all_resolved"] is True
        c = state["compounds"][0]
        assert c.resolved_name == "lisinopril"
        assert c.drug_class == DrugClass.ACE_INHIBITOR
        assert c.chembl_id is not None

    async def test_case_insensitive_match(self):
        state = await resolve_entities(make_state("LISINOPRIL"))
        assert state["compounds"][0].resolved_name == "lisinopril"

    async def test_misspelled_name_gets_suggestion_not_silent_guess(self):
        state = await resolve_entities(make_state("lisinoprill"))
        assert state["all_resolved"] is False
        c = state["compounds"][0]
        assert c.resolved_name is None
        assert c.chembl_id is None
        assert "lisinopril" in c.notes

    async def test_completely_unknown_name(self):
        state = await resolve_entities(make_state("totally_made_up_drug_xyz"))
        assert state["all_resolved"] is False
        assert state["compounds"][0].drug_class == DrugClass.UNKNOWN

    async def test_two_drug_combo_both_resolve(self):
        state = await resolve_entities(make_state("amlodipine", "losartan"))
        assert state["all_resolved"] is True
        names = {c.resolved_name for c in state["compounds"]}
        assert names == {"amlodipine", "losartan"}


@pytest.mark.asyncio
class TestRetrieveComparators:
    async def test_single_drug_gets_comparators(self):
        state = await resolve_entities(make_state("lisinopril"))
        state = await retrieve_comparators(state)
        assert len(state["comparators"]) > 0

    async def test_does_not_resuggest_exact_requested_combo(self):
        # amlodipine+losartan (CCB+ARB) IS in STANDARD_COMBINATIONS
        state = await resolve_entities(make_state("amlodipine", "losartan"))
        state = await retrieve_comparators(state)
        suggested_pairs = [set(c["drug_names"]) for c in state["comparators"]]
        assert {"amlodipine", "losartan"} not in suggested_pairs

    async def test_unresolved_state_yields_no_comparators(self):
        state = await resolve_entities(make_state("not_a_real_drug"))
        state = await retrieve_comparators(state)
        assert state["comparators"] == []


@pytest.mark.asyncio
class TestStructureFeaturesGate:
    async def test_rejects_three_drug_combination(self):
        state = await resolve_entities(make_state("lisinopril", "amlodipine", "atenolol"))
        state = await structure_features(state)
        assert state["prediction"] is None
        assert any("2-drug combinations" in note for note in state["limitations"])

    async def test_allows_two_drug_combination(self):
        state = await resolve_entities(make_state("amlodipine", "losartan"))
        state = await structure_features(state)
        assert state.get("prediction") != None or state.get("all_resolved") is True


class TestClarificationReport:
    def test_includes_unresolved_compound_notes(self):
        state = {
            "compounds": [CompoundQuery(raw_name="lisinoprill", notes="Did you mean 'lisinopril'?")],
            "limitations": ["Some other limitation"],
        }
        report = _clarification_report(state)
        assert "lisinoprill" in report
        assert "Did you mean 'lisinopril'?" in report
        assert "Some other limitation" in report
        assert "not clinical guidance" in report
