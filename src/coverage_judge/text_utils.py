import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd


TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
}


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_company_name(value: object) -> str:
    return re.sub(r"\s+", " ", normalize_text(value)).casefold()


def canonicalize_url(value: object) -> str:
    raw = normalize_text(value)
    if not raw:
        return ""

    try:
        parts = urlsplit(raw)
        if not parts.scheme or not parts.netloc:
            return raw.rstrip("/")

        filtered_query = []
        for key, val in parse_qsl(parts.query, keep_blank_values=True):
            lowered = key.casefold()
            if lowered.startswith("utm_") or lowered in TRACKING_QUERY_KEYS:
                continue
            filtered_query.append((key, val))

        path = parts.path.rstrip("/") or "/"
        return urlunsplit(
            (
                parts.scheme.casefold(),
                parts.netloc.casefold(),
                path,
                urlencode(filtered_query, doseq=True),
                "",
            )
        )
    except ValueError:
        return raw.rstrip("/")
