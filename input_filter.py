import re
import json
import os

# Path to the filter configuration file in the current working directory
FILTER_CONFIG_PATH = os.path.join(os.getcwd(), "filter-config.json")

def load_filter_rules():
    """
    Load filter rules (list of dicts with 'name' and 'pattern') from FILTER_CONFIG_PATH.
    If the file does not exist, initialize it to an empty list.
    """
    # Ensure the file exists
    if not os.path.exists(FILTER_CONFIG_PATH):
        with open(FILTER_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return []

    try:
        with open(FILTER_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        # Malformed JSON: treat as no rules
        pass
    return []

class InputFilter:
    """
    Dynamically load regex patterns from filter-config.json and
    provide a method to check for sensitive content.
    """

    @classmethod
    def get_all_patterns(cls):
        """
        Compile and return a list of regex patterns defined in filter-config.json.
        """
        rules = load_filter_rules()
        patterns = []
        for rule in rules:
            pat = rule.get("pattern")
            if not pat:
                continue
            try:
                patterns.append(re.compile(pat))
            except re.error:
                # Skip invalid regex patterns
                continue
        return patterns

    @classmethod
    def contains_sensitive(cls, text: str) -> bool:
        # strip out non‐word prefix/suffix so \b will work
        cleaned = text.strip()
        for pat in cls.get_all_patterns():
            if pat.search(cleaned):
                return True
        return False
