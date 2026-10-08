"""Circuit identity: map the schedule's venue name to a stable lowercase id.

Different layouts at the same venue (such as the 2020 Bahrain outer circuit) share one id; that is
a deliberate simplification. Kept free of fastf1 imports so it is cheap to import and test.
"""

import unicodedata

from ml.ingest.paths import slugify

# The schedule spells some venues differently across years; map them to one id.
CIRCUIT_ALIASES = {
    "monte_carlo": "monaco",
    "miami_gardens": "miami",
    "yas_island": "yas_marina",
    "singapore": "marina_bay",
}


def circuit_id(location: str) -> str:
    """Stable lowercase ascii id for a venue, e.g. 'Spa-Francorchamps' -> 'spa_francorchamps'."""
    ascii_name = unicodedata.normalize("NFKD", str(location)).encode("ascii", "ignore").decode()
    slug = slugify(ascii_name)
    return CIRCUIT_ALIASES.get(slug, slug)
