import pytest
from typer.testing import CliRunner

from scopeready import __description__, __title__, __version__
from scopeready.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_help_names_the_tool(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert __title__ in result.output
    assert __description__ in result.output


def test_version_prints_the_installed_version(runner: CliRunner) -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == __version__
