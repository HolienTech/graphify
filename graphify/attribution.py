"""Model attribution — which model actually produced a semantic result.

Graphify's semantic pass is LLM-backed: the entities and relationships that land
in ``graph.json`` are a model's judgment, and downstream systems query that graph
as fact. When a run mixes models — a chunk-level fallback, a resumed run, a
config change between builds — a single label on the artifact would be a lie.
So this module carries the *set* of models a run used, and for each one whether
the provider told us what it served or we only know what we asked for.

Two sources, and the distinction is deliberately not collapsed:

``REPORTED``
    The provider echoed back the model id it actually served. Authoritative.

``REQUESTED``
    The provider was silent about what it ran, so this is the id we asked for.
    A run that silently fell back to a different model would still read as the
    requested id here — which is exactly why it is labelled differently.

Attribution never comes from asking a model what it is. That is unreliable;
generated text is not evidence. It comes from the provider response, or failing
that from the resolved config.

Absent attribution reads as "unknown", never as a default model id. A
``graph.json`` written before this module existed carries no model field, and
"unknown" is the honest answer for it.

Everything here is fail-open and total: it accepts malformed input and returns
something usable rather than raising. Attribution must never break a build.
"""

from __future__ import annotations

# Where a model id came from. Order matters for display: the authoritative
# source is listed first when one model has both.
REPORTED = "reported"
REQUESTED = "requested"

_SOURCE_ORDER = {REPORTED: 0, REQUESTED: 1}

#: Rendered when nothing is known. Never substitute a default model id here —
#: "we did not record it" and "it was the default" are different claims.
UNKNOWN = "unknown"

# The key attribution travels under, both on in-flight result dicts and in the
# artifacts on disk.
KEY = "models"


def _clean(value: object) -> str:
    """Coerce a model id to a trimmed string, or "" if it is not usable."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def record(model: object, source: str = REQUESTED) -> dict | None:
    """Build one ``{"model": ..., "source": ...}`` record.

    Returns ``None`` for an unusable model id, so callers can drop it rather
    than record a blank or a ``None`` masquerading as a model name.
    """
    name = _clean(model)
    if not name:
        return None
    if source not in _SOURCE_ORDER:
        source = REQUESTED
    return {"model": name, "source": source}


def attach(result: dict, requested: object, served: object = None) -> dict:
    """Record on ``result`` which model produced it, and return ``result``.

    ``served`` is what the provider said it actually ran; ``requested`` is what
    we asked for. Where the provider reports what it served we prefer that and
    mark it ``REPORTED``; where it does not we fall back to the requested value
    and mark it ``REQUESTED``.

    The flat ``model`` key is still written, unchanged, for every existing
    caller that reads it.
    """
    if not isinstance(result, dict):
        return result

    rec = record(served, REPORTED) or record(requested, REQUESTED)
    if rec is None:
        return result

    result["model"] = rec["model"]
    result["model_source"] = rec["source"]
    result[KEY] = [dict(rec)]
    return result


def from_result(result: object) -> list[dict]:
    """Read attribution off an in-flight result dict.

    Prefers an already-merged ``models`` list; falls back to the flat
    ``model`` / ``model_source`` pair that a single provider call writes. A
    result with neither yields ``[]`` — unknown.
    """
    if not isinstance(result, dict):
        return []

    existing = result.get(KEY)
    if isinstance(existing, list) and existing:
        return normalize(existing)

    rec = record(result.get("model"), result.get("model_source") or REQUESTED)
    return [rec] if rec else []


def from_artifact(data: object) -> list[dict]:
    """Read attribution back out of a loaded artifact (``graph.json`` etc.).

    An artifact written before attribution existed has no such field, and this
    returns ``[]`` for it — which renders as "unknown". It must never fall back
    to a default model id: absent is not the same as default.
    """
    if not isinstance(data, dict):
        return []

    # `models` may sit at the top level or under an additive metadata block,
    # depending on the artifact. Check both; neither is required to exist.
    for container in (data, data.get("metadata"), data.get("semantic")):
        if isinstance(container, dict):
            found = container.get(KEY)
            if isinstance(found, list) and found:
                return normalize(found)
    return []


def normalize(records: object) -> list[dict]:
    """Coerce a raw list into clean, deduplicated records, order preserved.

    Tolerates plain strings (treated as ``REQUESTED``, since a bare name
    carries no evidence that the provider confirmed it) and drops anything
    unusable.
    """
    if not isinstance(records, (list, tuple)):
        return []

    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in records:
        if isinstance(item, dict):
            rec = record(item.get("model"), item.get("source") or REQUESTED)
        else:
            rec = record(item, REQUESTED)
        if rec is None:
            continue
        key = (rec["model"], rec["source"])
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def merge(*sources: object) -> list[dict]:
    """Union attribution across chunks, retries and resumed runs.

    This is the whole point of carrying a set: if two chunks were answered by
    different models, both are recorded. Collapsing them to the first would
    describe a mixed run as a single-model run.

    Accepts result dicts, record lists, or a mix. Order of first appearance is
    preserved so the label is stable across runs.
    """
    collected: list[dict] = []
    for src in sources:
        if isinstance(src, dict):
            collected.extend(from_result(src))
        elif isinstance(src, (list, tuple)):
            collected.extend(normalize(src))
    return normalize(collected)


def merge_into(records: object, result: object) -> list[dict]:
    """Fold one result's attribution into a running list. Returns the union."""
    return merge(records if records is not None else [], result)


def format_models(records: object) -> str:
    """Render attribution for humans.

    ``[]`` renders as "unknown". Each model is annotated with where its id came
    from, and a model seen both ways shows both — the ambiguity is real and
    hiding it would be the lie this module exists to prevent.
    """
    recs = normalize(records)
    if not recs:
        return UNKNOWN

    grouped: dict[str, list[str]] = {}
    for rec in recs:
        grouped.setdefault(rec["model"], []).append(rec["source"])

    parts = []
    for model, sources in grouped.items():
        ordered = sorted(set(sources), key=lambda s: _SOURCE_ORDER.get(s, 99))
        parts.append(f"{model} ({'/'.join(ordered)})")
    return ", ".join(parts)


def model_names(records: object) -> list[str]:
    """Just the distinct model ids, order preserved. ``[]`` when unknown."""
    out: list[str] = []
    for rec in normalize(records):
        if rec["model"] not in out:
            out.append(rec["model"])
    return out


def is_mixed(records: object) -> bool:
    """True when more than one distinct model produced the run."""
    return len(model_names(records)) > 1
