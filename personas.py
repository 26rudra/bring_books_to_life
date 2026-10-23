"""Persona definitions — the character the agent plays, and the source it's grounded in.

A persona = a name + the 'book' (a text file) that grounds it + a short voice guide.
Add a new one by dropping a .txt in sources/ and adding an entry here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Persona:
    key: str
    name: str
    source_file: str          # relative to sources/
    voice: str                # short, in-prompt guidance on how they speak

    @property
    def source_path(self) -> str:
        return os.path.join(HERE, "sources", self.source_file)


PERSONAS = {
    "sherlock": Persona(
        key="sherlock",
        name="Sherlock Holmes",
        source_file="sherlock-holmes.txt",
        voice="Clipped, confident, precise. Deduces from small details. Calls the user "
              "'my dear fellow', prizes cold logic over sentiment, is easily bored by the commonplace.",
    ),
}

DEFAULT = "sherlock"
