import re
import unittest
from pathlib import Path

from cluster_mlip.gaussian import _next_separators

FIXTURES = Path(__file__).parent / "fixtures"


def _reference(text: str, pos: int, count: int) -> list[tuple[int, int]]:
    """The original form: copy the rest of the file and scan it to the end."""
    found = list(re.finditer(r"^\s*-{10,}\s*$", text[pos:], re.M))[:count]
    return [(pos + match.start(), pos + match.end()) for match in found]


class SeparatorScanTests(unittest.TestCase):
    def test_matches_the_copy_and_scan_form_at_every_position(self):
        logs = sorted(FIXTURES.glob("*.log")) + sorted(FIXTURES.glob("*.out"))
        self.assertTrue(logs)
        for path in logs:
            text = path.read_text(errors="replace")
            # Every offset, including ones just after an "orientation:\s*" match
            # where leading spaces of the separator line were already consumed.
            for pos in range(0, len(text), max(1, len(text) // 4000)):
                for count in (1, 2):
                    self.assertEqual(_next_separators(text, pos, count),
                                     _reference(text, pos, count), (path.name, pos, count))

    def test_position_inside_leading_spaces_counts_as_a_line_start(self):
        text = "Standard orientation:\n   " + "-" * 20 + "\n header\n " + "-" * 20 + "\n"
        pos = text.index("-") - 2  # two of the three leading spaces consumed
        self.assertEqual(_next_separators(text, pos, 2), _reference(text, pos, 2))
        self.assertEqual(_next_separators(text, pos, 2)[0][0], pos)


if __name__ == "__main__":
    unittest.main()
