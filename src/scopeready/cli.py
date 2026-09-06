"""Command line entry point.

Commands are added by the step that makes them work: a command that prints a
plausible answer built from nothing is worse than a missing one, because only
the first kind gets believed.
"""

import typer

from scopeready import __description__, __title__, __version__

app = typer.Typer(name=__title__, help=__description__, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Group the commands under one help screen."""


@app.command()
def version() -> None:
    """Print the installed ScopeReady version."""
    typer.echo(__version__)
