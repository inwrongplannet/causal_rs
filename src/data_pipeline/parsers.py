import ast
import json

import pandas as pd


def parse_history(history_str: str) -> list[str]:
    if pd.isna(history_str):
        return []
    text = str(history_str).strip()
    if not text:
        return []
    return [tok for tok in text.split() if tok]


def parse_impressions(impressions_str: str) -> list[tuple[str, int]]:
    if pd.isna(impressions_str):
        return []
    pairs: list[tuple[str, int]] = []
    for token in str(impressions_str).split():
        if "-" not in token:
            continue
        item_id, click_flag = token.rsplit("-", 1)
        if not item_id:
            continue
        pairs.append((item_id, 1 if click_flag == "1" else 0))
    return pairs


def parse_entities(entity_blob: str) -> list[str]:
    if pd.isna(entity_blob):
        return []
    text = str(entity_blob).strip()
    if not text or text in {"[]", "nan", "None"}:
        return []
    payload = None
    for parser in (json.loads, ast.literal_eval):
        try:
            payload = parser(text)
            break
        except Exception:  # noqa: BLE001, S112
            continue
    if not isinstance(payload, list):
        return []
    ids: list[str] = []
    for node in payload:
        if isinstance(node, dict):
            entity_id = node.get("WikidataId") or node.get("WikidataID") or node.get("Label")
            if entity_id:
                ids.append(str(entity_id))
    return ids
