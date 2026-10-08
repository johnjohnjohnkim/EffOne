"""Team continuity across renames.

fastf1 gives a team a new TeamId whenever it is renamed, which would reset its form to "no history"
even though it is the same entry. This table maps each id to ONE id for the entry (the most recent
one for Aston Martin, RB and Alpine; the earlier `sauber` for the Sauber entry, since its ids go
sauber -> alfa -> sauber -> audi). It is a judgement from public knowledge, not something the data
states:

- Force India -> Racing Point -> Aston Martin
- Toro Rosso -> AlphaTauri -> RB (Racing Bulls)
- Renault -> Alpine
- Sauber -> Alfa Romeo -> Sauber (Kick Sauber) -> Audi (2026 takeover of the Sauber entry)

A new entry (Cadillac, 2026) has no lineage and so no history, which is correct. Note that rules
changes reset car performance regardless of the entry's continuity; the model has to cope with that.
"""

from collections.abc import Mapping
from types import MappingProxyType

TEAM_LINEAGE: Mapping[str, str] = MappingProxyType(
    {
        "force_india": "aston_martin",
        "racing_point": "aston_martin",
        "toro_rosso": "rb",
        "alphatauri": "rb",
        "renault": "alpine",
        "alfa": "sauber",
        "audi": "sauber",
    }
)


def team_key(team_id: str) -> str:
    """The id that identifies the same entry across renames."""
    return TEAM_LINEAGE.get(team_id, team_id)
