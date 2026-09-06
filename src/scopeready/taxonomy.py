"""The taxonomy: the rubric, kept as data the engine never knows by name.

The engine can name no category. It loads records, filters them by the gate and
asks one question per surviving record. That is what makes a vertical an extra
YAML file instead of a branch in the pipeline, and what makes one category one
metric — the unit the quality of this product is argued in.

Everything here is validated at load time, including the applicability
expressions. A rubric that parses on the twelfth minute of a run is a rubric
that fails on the twelfth minute of a run.
"""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from scopeready.applicability import (
    Expression,
    describe_granularity,
    granularity_applies,
    unknown_features,
)
from scopeready.models import (
    CategoryId,
    CategoryWeight,
    FeatureId,
    Granularity,
    NonEmptyStr,
    ProjectProfile,
    SkippedCategory,
    SkipReason,
    WeightSource,
)

PROFILE_FILE = "profile.yaml"
CATEGORIES_FILE = "gaps.client_agreement.yaml"

# Probe text is written for a model, and a one-line probe produces a one-line
# judgement. The floor is not style policing: it is the cheapest available
# check that a record was actually authored.
ProbeText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=20)]


class TaxonomyError(ValueError):
    """A rubric file that cannot be trusted to mean what it says."""


class SpecModel(BaseModel):
    # A misspelled key in a rubric file must be an error. Silently ignored, it
    # would mean a category whose `insufficient_when` never took effect, and
    # nothing in the output would say so.
    model_config = ConfigDict(frozen=True, extra="forbid")


class FeatureSpec(SpecModel):
    """One project property the gate switches categories on."""

    id: FeatureId
    title: NonEmptyStr
    # Asked of the model as a yes/no question about the corpus, so it has to be
    # answerable from documents rather than from knowing the company.
    question: ProbeText


class ProfileSpec(SpecModel):
    version: NonEmptyStr
    features: tuple[FeatureSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _feature_ids_are_unique(self) -> Self:
        _reject_duplicates(feature.id for feature in self.features)
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(feature.id for feature in self.features)


class CategorySpec(SpecModel):
    """One line of the rubric: what is asked, what counts, what it costs."""

    id: CategoryId
    title: NonEmptyStr
    # A list rather than one value: acceptance criteria are asked of a ticket
    # and of an epic, a warranty period only of a project.
    granularity: tuple[Granularity, ...] = Field(min_length=1)
    # Normalized to (0, 1] so that a severity threshold means the same thing for
    # every category. A 1-5 rubric would make the number readable and the
    # comparison meaningless.
    weight: CategoryWeight
    weight_source: WeightSource = WeightSource.SEED
    applies_when: NonEmptyStr
    probe: ProbeText
    covered_when: ProbeText
    # The most valuable field in the record: near misses are what separate a
    # useful `partial` from a rubric that only knows yes and no.
    insufficient_when: ProbeText
    # One topic is named differently in a ticket, a wiki page and a chat, and
    # the refutation pass searches the whole corpus with these.
    retrieval_queries: tuple[NonEmptyStr, ...] = Field(min_length=1)
    why_it_costs: ProbeText

    @model_validator(mode="after")
    def _granularity_is_unique(self) -> Self:
        _reject_duplicates(level.value for level in self.granularity)
        return self


class CategoriesSpec(SpecModel):
    version: NonEmptyStr
    categories: tuple[CategorySpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _category_ids_are_unique(self) -> Self:
        _reject_duplicates(category.id for category in self.categories)
        return self


class Taxonomy:
    """A loaded rubric with its expressions parsed and its version pinned."""

    def __init__(
        self,
        profile: ProfileSpec,
        categories: CategoriesSpec,
        expressions: Mapping[str, Expression],
        digest: str,
    ) -> None:
        self.profile = profile
        self.categories = categories.categories
        self.version = f"{profile.version}/{categories.version}"
        self._expressions = expressions
        self.digest = digest

    def expression(self, category_id: str) -> Expression:
        return self._expressions[category_id]

    @property
    def feature_ids(self) -> tuple[str, ...]:
        return self.profile.ids

    def __len__(self) -> int:
        return len(self.categories)


def load_taxonomy(directory: Path) -> Taxonomy:
    """Read, validate and fingerprint the rubric.

    The digest covers the raw bytes of both files. Two runs that differ only in
    a reworded probe are two different measurements, and the digest is what
    makes that visible in a report instead of arguable afterwards.
    """
    profile_bytes = _read(directory / PROFILE_FILE)
    categories_bytes = _read(directory / CATEGORIES_FILE)

    profile = _validate(ProfileSpec, profile_bytes, PROFILE_FILE)
    categories = _validate(CategoriesSpec, categories_bytes, CATEGORIES_FILE)

    expressions: dict[str, Expression] = {}
    for category in categories.categories:
        expression = Expression.parse(category.applies_when)
        missing = unknown_features(expression, profile.ids)
        if missing:
            msg = (
                f"category {category.id!r} refers to features that "
                f"{PROFILE_FILE} does not define: {', '.join(sorted(missing))}"
            )
            raise TaxonomyError(msg)
        expressions[category.id] = expression

    digest = hashlib.sha256(profile_bytes + categories_bytes).hexdigest()[:16]
    return Taxonomy(profile, categories, expressions, digest)


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        msg = f"cannot read taxonomy file {path}: {error}"
        raise TaxonomyError(msg) from error


def _validate[T: SpecModel](model: type[T], raw: bytes, name: str) -> T:
    # safe_load, never load: a rubric file must not be able to construct Python
    # objects through YAML tags. Same rule as the no-eval rule, one layer down.
    try:
        parsed: Any = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        msg = f"{name} is not valid YAML: {error}"
        raise TaxonomyError(msg) from error
    if not isinstance(parsed, dict):
        msg = f"{name} must contain a mapping at the top level"
        raise TaxonomyError(msg)
    try:
        return model.model_validate(parsed)
    except ValueError as error:
        msg = f"{name} is not a valid taxonomy file: {error}"
        raise TaxonomyError(msg) from error


def _reject_duplicates(values: Iterable[str]) -> None:
    seen: list[str] = list(values)
    duplicates = sorted({value for value in seen if seen.count(value) > 1})
    if duplicates:
        msg = f"duplicate identifiers: {', '.join(duplicates)}"
        raise ValueError(msg)


def categories_by_id(
    categories: Sequence[CategorySpec],
) -> Mapping[str, CategorySpec]:
    return {category.id: category for category in categories}


@dataclass(frozen=True, slots=True)
class Selection:
    """The categories a run will ask about, and the ones it will not."""

    applicable: tuple[CategorySpec, ...]
    skipped: tuple[SkippedCategory, ...]


def select_categories(
    taxonomy: Taxonomy, profile: ProjectProfile, unit: Granularity
) -> Selection:
    """Narrow the rubric to this project and this level of scope.

    Granularity is checked first and the search stops at the first failing
    filter, so each removal has exactly one stated reason. The order is fixed
    rather than natural: granularity is a property of how the run was launched,
    while applicability depends on a profile the model produced, and reporting
    the structural reason first keeps a mis-scoped run from looking like a bad
    profile.
    """
    applicable: list[CategorySpec] = []
    skipped: list[SkippedCategory] = []

    for category in taxonomy.categories:
        if not granularity_applies(category.granularity, unit):
            skipped.append(
                SkippedCategory(
                    category_id=category.id,
                    category_title=category.title,
                    reason=SkipReason.WRONG_GRANULARITY,
                    explanation=describe_granularity(category.granularity, unit),
                )
            )
            continue
        expression = taxonomy.expression(category.id)
        if not expression.evaluate(profile):
            skipped.append(
                SkippedCategory(
                    category_id=category.id,
                    category_title=category.title,
                    reason=SkipReason.NOT_APPLICABLE,
                    explanation=expression.explain(profile),
                )
            )
            continue
        applicable.append(category)

    return Selection(applicable=tuple(applicable), skipped=tuple(skipped))
