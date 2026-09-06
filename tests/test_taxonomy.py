"""Tests for the rubric and the gate that narrows it.

The rubric is data, so most of these check that a bad file is rejected rather
than absorbed. An accepted bad rubric produces a plausible report, which is the
expensive kind of wrong.
"""

from pathlib import Path

import pytest

from scopeready.applicability import (
    Expression,
    ExpressionError,
    profile_from_flags,
    unknown_features,
)
from scopeready.config import TAXONOMY_DIR
from scopeready.models import Granularity, SkipReason
from scopeready.taxonomy import (
    Taxonomy,
    TaxonomyError,
    load_taxonomy,
    select_categories,
)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_DIR)


# --- the expression language ------------------------------------------------


@pytest.mark.parametrize(
    ("source", "flags", "expected"),
    [
        ("true", {}, True),
        ("false", {}, False),
        ("has_payments", {"has_payments": True}, True),
        ("has_payments", {}, False),
        ("not has_payments", {}, True),
        ("a and b", {"a": True, "b": False}, False),
        ("a or b", {"a": True, "b": False}, True),
        ("a and (b or not c)", {"a": True, "c": True}, False),
        ("a and (b or not c)", {"a": True, "b": True, "c": True}, True),
    ],
)
def test_expressions_evaluate_against_a_profile(
    source: str, flags: dict[str, bool], expected: bool
) -> None:
    assert Expression.parse(source).evaluate(profile_from_flags(flags)) is expected


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('rm -rf /')",
        "().__class__.__mro__",
        "[x for x in range(3)]",
        "has_payments == True",
        "open('/etc/passwd').read()",
        "lambda: True",
        "has_payments if x else y",
        "1 + 1",
        "f'{has_payments}'",
        "has_payments; drop_everything()",
    ],
)
def test_anything_beyond_boolean_logic_is_refused(source: str) -> None:
    with pytest.raises(ExpressionError):
        Expression.parse(source)


def test_a_non_boolean_constant_is_refused() -> None:
    with pytest.raises(ExpressionError, match="only true and false"):
        Expression.parse("1")


def test_an_empty_expression_is_refused() -> None:
    with pytest.raises(ExpressionError, match="must not be empty"):
        Expression.parse("   ")


def test_literals_are_not_mistaken_for_features() -> None:
    assert Expression.parse("true or has_payments").feature_names == {"has_payments"}


def test_unknown_features_are_reported_against_the_profile() -> None:
    expression = Expression.parse("has_payments and has_typo")
    assert unknown_features(expression, ["has_payments"]) == {"has_typo"}


def test_an_explanation_names_the_values_the_verdict_rests_on() -> None:
    profile = profile_from_flags({"a": True})
    assert Expression.parse("a").explain(profile) == "a=true"
    assert Expression.parse("a and b").explain(profile) == "a and b (a=true, b=false)"


# --- loading the shipped rubric ---------------------------------------------


def test_the_shipped_rubric_loads(taxonomy: Taxonomy) -> None:
    assert len(taxonomy) == 19
    assert len(taxonomy.feature_ids) == 12


def test_weights_are_normalized(taxonomy: Taxonomy) -> None:
    # A severity threshold has to mean the same thing for every category, which
    # is only true while every weight lives on the same scale.
    assert all(0.0 < category.weight <= 1.0 for category in taxonomy.categories)


def test_the_digest_changes_with_the_files(tmp_path: Path, taxonomy: Taxonomy) -> None:
    for name in ("profile.yaml", "gaps.client_agreement.yaml"):
        (tmp_path / name).write_bytes((TAXONOMY_DIR / name).read_bytes())
    assert load_taxonomy(tmp_path).digest == taxonomy.digest

    path = tmp_path / "gaps.client_agreement.yaml"
    path.write_text(path.read_text().replace("weight: 0.9", "weight: 0.8", 1))
    assert load_taxonomy(tmp_path).digest != taxonomy.digest


def _write(directory: Path, categories: str) -> None:
    (directory / "profile.yaml").write_text(
        'version: "1"\n'
        "features:\n"
        "  - id: has_payments\n"
        "    title: Handles money\n"
        "    question: Does the project take or pay out money in any form?\n"
    )
    (directory / "gaps.client_agreement.yaml").write_text(categories)


_CATEGORY = """version: "1"
categories:
  - id: payments_pci
    title: Payments
    granularity: [project]
    weight: 0.8
    applies_when: "{applies_when}"
    probe: Does the corpus state how money moves through the system?
    covered_when: A provider is named together with the failure cases.
    insufficient_when: A provider is named with nothing about refunds.
    retrieval_queries:
      - payment provider refund
    why_it_costs: Payment edge cases are where money is actually lost.
"""


def test_a_typo_in_applies_when_fails_at_load(tmp_path: Path) -> None:
    # Left to run time it would read as "the feature is false", silently
    # removing the category — which looks exactly like the gate working.
    _write(tmp_path, _CATEGORY.format(applies_when="has_paymnets"))
    with pytest.raises(TaxonomyError, match="does not define"):
        load_taxonomy(tmp_path)


def test_a_misspelled_key_is_not_ignored(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _CATEGORY.format(applies_when="true").replace(
            "insufficient_when:", "insuficient_when:"
        ),
    )
    with pytest.raises(TaxonomyError):
        load_taxonomy(tmp_path)


def test_a_duplicated_category_is_refused(tmp_path: Path) -> None:
    body = _CATEGORY.format(applies_when="true")
    _write(tmp_path, body + body.split("categories:\n")[1])
    with pytest.raises(TaxonomyError, match="duplicate"):
        load_taxonomy(tmp_path)


def test_a_yaml_tag_cannot_construct_objects(tmp_path: Path) -> None:
    _write(tmp_path, "!!python/object/apply:os.system ['echo owned']\n")
    with pytest.raises(TaxonomyError):
        load_taxonomy(tmp_path)


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(TaxonomyError, match="cannot read"):
        load_taxonomy(tmp_path)


# --- the gate ---------------------------------------------------------------


def test_the_gate_removes_categories_and_says_why(taxonomy: Taxonomy) -> None:
    profile = profile_from_flags(
        {
            "has_multiple_roles": True,
            "has_personal_data": True,
            "has_end_user_interface": True,
        }
    )
    selection = select_categories(taxonomy, profile, Granularity.EPIC)
    asked = {category.id for category in selection.applicable}
    reasons = {skipped.category_id: skipped for skipped in selection.skipped}

    assert "roles_and_permissions" in asked
    assert "data_lifecycle_gdpr" in asked
    # The whole point of the gate: no payment gap on a project that takes no
    # money, and no console certification gap outside a game.
    assert reasons["payments_pci"].reason is SkipReason.NOT_APPLICABLE
    assert "has_payments=false" in reasons["payments_pci"].explanation
    assert "liveops_and_monetization" in reasons

    assert len(selection.applicable) + len(selection.skipped) == len(taxonomy)


def test_a_game_on_console_unlocks_the_vertical_without_code(
    taxonomy: Taxonomy,
) -> None:
    profile = profile_from_flags(
        {"is_game_project": True, "targets_console_platforms": True}
    )
    asked = {
        category.id
        for category in select_categories(
            taxonomy, profile, Granularity.PROJECT
        ).applicable
    }
    assert {"platform_certification", "age_rating", "game_engine_and_version"} <= asked


def test_granularity_is_reported_separately_from_applicability(
    taxonomy: Taxonomy,
) -> None:
    selection = select_categories(taxonomy, profile_from_flags({}), Granularity.TICKET)
    reasons = {skipped.category_id: skipped for skipped in selection.skipped}
    assert reasons["warranty_period"].reason is SkipReason.WRONG_GRANULARITY
    assert "acceptance_criteria" in {category.id for category in selection.applicable}
