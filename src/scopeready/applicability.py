"""The applicability gate: which taxonomy categories a project can be judged by.

Without this the engine raises a PCI gap on a project that takes no money and a
console certification gap in a web admin panel. False positives kill a tool like
this faster than incomplete recall does, because the second costs a missed
question and the first costs the reader's trust in every question.

The expression language is deliberately tiny: feature names, `and`, `or`, `not`
and parentheses. It is parsed with `ast` against an allowlist of node types and
walked by this module. Nothing here calls `eval`, and the taxonomy is read with
`yaml.safe_load`, because a rubric file is data that ships next to the code and
would otherwise be a way to execute code by editing a config.
"""

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Self

from scopeready.models import Granularity, ProjectProfile

# `Load` is the context every readable Name carries; without it a bare feature
# name does not parse. `Constant` is allowed for `true`/`false` so a category
# that applies everywhere can say so instead of inventing a feature that is
# always set.
_ALLOWED_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Name,
        ast.Load,
        ast.Constant,
    }
)


# The rubric is authored in YAML, where a boolean is spelled `true`, but Python's
# parser reads that as a name and reserves the capitalized spelling. Rather than
# rewriting the expression text before parsing — which would edit a string the
# error messages then quote back — the two spellings are resolved here. A
# profile feature may therefore not be named `true` or `false`, which costs
# nothing and is worth the absence of a preprocessing step.
_LITERALS: Mapping[str, bool] = {"true": True, "false": False}


class ExpressionError(ValueError):
    """An `applies_when` expression that cannot be trusted to mean anything."""


@dataclass(frozen=True, slots=True)
class Expression:
    """A parsed, allowlisted boolean expression over profile features."""

    source: str
    feature_names: frozenset[str]
    _tree: ast.Expression

    @classmethod
    def parse(cls, source: str) -> Self:
        text = source.strip()
        if not text:
            msg = "an applicability expression must not be empty"
            raise ExpressionError(msg)
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as error:
            msg = f"cannot parse {source!r}: {error.msg}"
            raise ExpressionError(msg) from error

        names: set[str] = set()
        for node in ast.walk(tree):
            if type(node) not in _ALLOWED_NODES:
                msg = (
                    f"{type(node).__name__} is not allowed in an applicability "
                    f"expression: {source!r}"
                )
                raise ExpressionError(msg)
            if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
                msg = f"only true and false are allowed as constants: {source!r}"
                raise ExpressionError(msg)
            if isinstance(node, ast.Name) and node.id not in _LITERALS:
                names.add(node.id)
        return cls(source=text, feature_names=frozenset(names), _tree=tree)

    def evaluate(self, profile: ProjectProfile) -> bool:
        return _evaluate(self._tree.body, profile)

    def explain(self, profile: ProjectProfile) -> str:
        """State the feature values the verdict rests on.

        A gate that subtracts a category silently is indistinguishable from a
        category missing from the file, so every removal has to be able to name
        the reason it happened.
        """
        if not self.feature_names:
            return self.source
        settings = ", ".join(
            f"{name}={str(profile.is_set(name)).lower()}"
            for name in sorted(self.feature_names)
        )
        # A single-name expression is its own explanation; repeating it as
        # "has_payments (has_payments=false)" adds noise to the section a reader
        # goes to precisely when something looks wrong.
        if self.source in self.feature_names:
            return settings
        return f"{self.source} ({settings})"


def _evaluate(node: ast.expr, profile: ProjectProfile) -> bool:
    match node:
        case ast.Constant(value=bool() as value):
            return value
        case ast.Name(id=name) if name in _LITERALS:
            return _LITERALS[name]
        case ast.Name(id=name):
            return profile.is_set(name)
        case ast.UnaryOp(op=ast.Not(), operand=operand):
            return not _evaluate(operand, profile)
        case ast.BoolOp(op=ast.And(), values=values):
            return all(_evaluate(value, profile) for value in values)
        case ast.BoolOp(op=ast.Or(), values=values):
            return any(_evaluate(value, profile) for value in values)
        case _:  # pragma: no cover - Expression.parse rejects everything else
            msg = f"unexpected node {type(node).__name__} survived parsing"
            raise ExpressionError(msg)


def unknown_features(expression: Expression, known: Iterable[str]) -> frozenset[str]:
    """Feature names the expression uses that the profile file does not define.

    Resolved at taxonomy load time rather than at run time on purpose: a typo in
    `applies_when` otherwise reads as "the feature is false", which removes the
    category, which looks exactly like the gate working. The run would finish,
    the report would look clean, and the category would be gone.
    """
    return frozenset(expression.feature_names) - frozenset(known)


def granularity_applies(levels: Iterable[Granularity], unit: Granularity) -> bool:
    return unit in tuple(levels)


def describe_granularity(levels: Iterable[Granularity], unit: Granularity) -> str:
    stated = ", ".join(level.value for level in levels)
    return f"category applies at {stated}; the unit is {unit.value}"


def profile_from_flags(flags: Mapping[str, bool]) -> ProjectProfile:
    """Build a profile from plain booleans, for the gate table and for tests."""
    from scopeready.models import ProfileFeature

    return ProjectProfile(
        features={
            name: ProfileFeature(value=value, rationale="set by hand")
            for name, value in flags.items()
        }
    )
