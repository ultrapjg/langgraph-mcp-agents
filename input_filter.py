import re
import os
import requests

# Base URL for the filter management API
FILTER_API_URL = os.getenv("FILTER_API_URL", "http://localhost:8000")

def load_filter_rules():
    """Load filter rules from the backend API."""
    try:
        resp = requests.get(f"{FILTER_API_URL}/filters", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []

def save_filter_rules(rules):
    """Persist filter rules through the backend API."""
    try:
        resp = requests.put(f"{FILTER_API_URL}/filters", json=rules, timeout=5)
        resp.raise_for_status()
        return True
    except Exception:
        return False

def delete_filter_rule(rule_id):
    """Delete a filter rule via the backend API."""
    try:
        resp = requests.delete(f"{FILTER_API_URL}/filters/{rule_id}", timeout=5)
        resp.raise_for_status()
        return True
    except Exception:
        return False

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
