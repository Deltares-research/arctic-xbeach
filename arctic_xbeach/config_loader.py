"""Loading of a run's config.yaml. PyYAML keeps the last of two equal keys without a warning (a published run had two
`thermal:` blocks and used the second), so a duplicate key is an error here."""
from pathlib import Path

import yaml


class UniqueKeySafeLoader(yaml.SafeLoader):
    """yaml.SafeLoader that raises a ValueError when a mapping repeats a key."""

    def construct_mapping(self, node, deep=False):
        first_line = {}
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                if key in first_line:
                    raise ValueError(f"Duplicate key '{key}' in config (line {key_node.start_mark.line + 1}, "
                                     f"first used on line {first_line[key]}); remove one of them")
            except TypeError:  # unhashable key: left to the parent, which reports it
                continue
            first_line[key] = key_node.start_mark.line + 1
        return super().construct_mapping(node, deep=deep)


def load_config(path_or_text) -> dict:
    """Parse a config.yaml (Path) or its text with UniqueKeySafeLoader."""
    text = path_or_text.read_text() if isinstance(path_or_text, Path) else path_or_text
    return yaml.load(text, Loader=UniqueKeySafeLoader)
