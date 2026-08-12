"""Which model asserted this edge?

The graph is a durable artifact other systems query as fact. These tests pin
the three cases that make the answer trustworthy: a run that mixed models must
say so, a run that recorded nothing must say "unknown" rather than name a
default, and a graph written before attribution existed must still load.
"""

import json
import tempfile
from pathlib import Path

import pytest

from graphify import attribution
from graphify.build import build_from_json
from graphify.cluster import cluster
from graphify.export import to_json
from graphify.llm import _merge_into
from graphify.report import generate

FIXTURES = Path(__file__).parent / "fixtures"


def make_graph():
    return build_from_json(json.loads((FIXTURES / "extraction.json").read_text()))


# ── records and sources ──────────────────────────────────────────────────────

def test_reported_and_requested_are_distinct():
    """A confirmed model and an assumed one must never read the same."""
    served = attribution.attach({}, "gemini-2.5-pro", "gemini-2.5-flash")
    assumed = attribution.attach({}, "gemini-2.5-pro")

    assert served["model"] == "gemini-2.5-flash"
    assert served["model_source"] == attribution.REPORTED
    assert assumed["model"] == "gemini-2.5-pro"
    assert assumed["model_source"] == attribution.REQUESTED

    assert "reported" in attribution.format_models(attribution.from_result(served))
    assert "requested" in attribution.format_models(attribution.from_result(assumed))


def test_served_model_wins_over_requested():
    """Where the provider says what it ran, that is what gets recorded.

    This is the silent-fallback case: we asked for pro and got flash.
    """
    result = attribution.attach({}, "gemini-2.5-pro", "gemini-2.5-flash")
    assert attribution.model_names(attribution.from_result(result)) == ["gemini-2.5-flash"]


def test_blank_served_falls_back_to_requested():
    for empty in (None, "", "   "):
        result = attribution.attach({}, "gemini-2.5-pro", empty)
        assert result["model"] == "gemini-2.5-pro"
        assert result["model_source"] == attribution.REQUESTED


# ── the mixed-model run ──────────────────────────────────────────────────────

def test_merge_into_unions_models_across_chunks():
    """A mixed run records the SET. Labelling it with the first chunk's model
    would misreport every other chunk's judgment."""
    merged = {
        "nodes": [], "edges": [], "hyperedges": [],
        "input_tokens": 0, "output_tokens": 0,
        attribution.KEY: [],
    }
    _merge_into(merged, attribution.attach(
        {"nodes": [{"id": "a"}], "input_tokens": 5, "output_tokens": 1},
        "gemini-2.5-pro", "gemini-2.5-pro"))
    _merge_into(merged, attribution.attach(
        {"nodes": [{"id": "b"}], "input_tokens": 5, "output_tokens": 1},
        "gemini-2.5-pro", "gemini-2.5-flash"))

    assert attribution.model_names(merged[attribution.KEY]) == [
        "gemini-2.5-pro", "gemini-2.5-flash"
    ]
    assert attribution.is_mixed(merged[attribution.KEY])
    # The nodes and tokens still merged normally.
    assert len(merged["nodes"]) == 2
    assert merged["input_tokens"] == 10


def test_merge_into_deduplicates_the_common_case():
    """Every chunk on one model yields one label, not N copies of it."""
    merged = {"nodes": [], "edges": [], "hyperedges": [],
              "input_tokens": 0, "output_tokens": 0, attribution.KEY: []}
    for _ in range(5):
        _merge_into(merged, attribution.attach({}, "gpt-5", "gpt-5"))

    assert merged[attribution.KEY] == [
        {"model": "gpt-5", "source": attribution.REPORTED}
    ]
    assert not attribution.is_mixed(merged[attribution.KEY])


def test_mixed_run_label_names_every_model():
    label = attribution.format_models([
        {"model": "gemini-2.5-pro", "source": attribution.REPORTED},
        {"model": "gemini-2.5-flash", "source": attribution.REPORTED},
    ])
    assert "gemini-2.5-pro" in label and "gemini-2.5-flash" in label


def test_same_model_seen_both_ways_keeps_both_sources():
    """Some chunks confirmed, some only assumed — that ambiguity is real."""
    recs = attribution.merge(
        [{"model": "gpt-5", "source": attribution.REPORTED}],
        [{"model": "gpt-5", "source": attribution.REQUESTED}],
    )
    assert len(recs) == 2
    label = attribution.format_models(recs)
    assert "reported" in label and "requested" in label
    # One model, so it is not a "mixed" run even though the sources differ.
    assert not attribution.is_mixed(recs)


# ── the unknown / absent case ────────────────────────────────────────────────

def test_absent_attribution_reads_as_unknown():
    """Never a default model id — absent and default are different claims."""
    assert attribution.format_models([]) == "unknown"
    assert attribution.format_models(None) == "unknown"
    assert attribution.from_result({}) == []
    assert attribution.model_names([]) == []


def test_result_without_any_model_yields_nothing():
    result = {"nodes": [], "edges": [], "input_tokens": 0, "output_tokens": 0}
    assert attribution.from_result(result) == []
    assert attribution.format_models(attribution.from_result(result)) == "unknown"


def test_unusable_model_ids_are_dropped_not_recorded():
    for junk in (None, "", "   ", 42, {}, []):
        assert attribution.record(junk) is None
    assert attribution.normalize([None, "", "gpt-5", 7]) == [
        {"model": "gpt-5", "source": attribution.REQUESTED}
    ]


def test_a_failed_run_records_no_model():
    """No chunk succeeded, so no model asserted anything."""
    merged = {"nodes": [], "edges": [], "hyperedges": [],
              "input_tokens": 0, "output_tokens": 0, attribution.KEY: []}
    assert attribution.format_models(merged[attribution.KEY]) == "unknown"


# ── old-artifact compatibility ───────────────────────────────────────────────

def test_old_graph_json_without_models_still_loads():
    """A graph written before attribution existed must load and read unknown."""
    old = {
        "nodes": [{"id": "a", "label": "A"}],
        "links": [],
        "hyperedges": [],
        "built_at_commit": "deadbeef",
    }
    assert attribution.from_artifact(old) == []
    assert attribution.format_models(attribution.from_artifact(old)) == "unknown"


def test_old_artifact_never_infers_a_default_model():
    """The regression that matters: absent must not become a model id."""
    label = attribution.format_models(attribution.from_artifact({"nodes": []}))
    assert label == "unknown"
    assert "gemini" not in label and "claude" not in label and "gpt" not in label


def test_from_artifact_tolerates_garbage():
    for junk in (None, [], "nope", 5, {"models": "not-a-list"}, {"models": []}):
        assert attribution.from_artifact(junk) == []


def test_from_artifact_reads_a_new_graph():
    new = {"nodes": [], "models": [{"model": "gemini-2.5-pro", "source": "reported"}]}
    assert attribution.model_names(attribution.from_artifact(new)) == ["gemini-2.5-pro"]


def test_old_cost_json_run_without_models_is_readable():
    """cost.json gained a per-run `models` key; old runs simply lack it."""
    old_run = {"date": "2026-01-01T00:00:00+00:00", "input_tokens": 10,
               "output_tokens": 5, "files": 3}
    assert attribution.format_models(old_run.get("models")) == "unknown"


# ── graph.json ───────────────────────────────────────────────────────────────

def test_to_json_records_the_models():
    G = make_graph()
    communities = cluster(G)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        to_json(G, communities, str(out), models=[
            {"model": "gemini-2.5-pro", "source": attribution.REPORTED}])
        data = json.loads(out.read_text())

    assert data["models"] == [{"model": "gemini-2.5-pro", "source": "reported"}]
    assert attribution.model_names(attribution.from_artifact(data)) == ["gemini-2.5-pro"]


def test_to_json_records_a_mixed_run_as_a_set():
    G = make_graph()
    communities = cluster(G)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        to_json(G, communities, str(out), models=[
            {"model": "gemini-2.5-pro", "source": attribution.REPORTED},
            {"model": "gemini-2.5-flash", "source": attribution.REPORTED},
        ])
        data = json.loads(out.read_text())

    assert attribution.model_names(attribution.from_artifact(data)) == [
        "gemini-2.5-pro", "gemini-2.5-flash"]


def test_to_json_omits_models_when_there_are_none():
    """An AST-only graph names no model rather than claiming a default."""
    G = make_graph()
    communities = cluster(G)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        to_json(G, communities, str(out))
        data = json.loads(out.read_text())

    assert "models" not in data
    assert attribution.from_artifact(data) == []


def test_to_json_survives_malformed_models():
    """Fail-open: a bad label must not cost you the graph."""
    G = make_graph()
    communities = cluster(G)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        assert to_json(G, communities, str(out), models="not-a-list") is True
        data = json.loads(out.read_text())

    assert "models" not in data
    assert len(data["nodes"]) > 0


# ── GRAPH_REPORT.md ──────────────────────────────────────────────────────────

def _report(models):
    from graphify.analyze import god_nodes, surprising_connections
    from graphify.cluster import score_all
    extraction = json.loads((FIXTURES / "extraction.json").read_text())
    G = build_from_json(extraction)
    communities = cluster(G)
    return generate(
        G, communities, score_all(G, communities),
        {cid: f"Community {cid}" for cid in communities},
        god_nodes(G), surprising_connections(G),
        {"total_files": 4, "total_words": 62400, "needs_graph": True, "warning": None},
        {"input": 10, "output": 5}, "./project", models=models,
    )


def test_report_names_the_semantic_model():
    assert "- Semantic model: gemini-2.5-pro (reported)" in _report(
        [{"model": "gemini-2.5-pro", "source": attribution.REPORTED}])


def test_report_says_unknown_when_nothing_was_recorded():
    assert "- Semantic model: unknown" in _report(None)


def test_report_flags_a_mixed_run():
    report = _report([
        {"model": "gemini-2.5-pro", "source": attribution.REPORTED},
        {"model": "gemini-2.5-flash", "source": attribution.REPORTED},
    ])
    assert "mixed run" in report
    assert "gemini-2.5-pro" in report and "gemini-2.5-flash" in report


def test_report_marks_an_unconfirmed_model_as_requested():
    assert "(requested)" in _report(
        [{"model": "gemini-2.5-pro", "source": attribution.REQUESTED}])


def test_report_survives_malformed_models():
    """A report without a model beats no report at all."""
    assert "- Semantic model: unknown" in _report("garbage")


# ── the claude-cli path, which reads what the provider served ────────────────

def test_claude_cli_preserves_the_model_usage_envelope():
    """modelUsage is the most authoritative signal on this path — keep it."""
    from graphify.llm import _call_claude_cli

    envelope = {
        "result": json.dumps({"nodes": [], "edges": [], "hyperedges": []}),
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "modelUsage": {"claude-sonnet-4-5-20250929": {"inputTokens": 10}},
        "stop_reason": "end_turn",
    }
    result = _run_claude_cli_with(envelope, _call_claude_cli)

    assert result["model"] == "claude-sonnet-4-5-20250929"
    assert result["model_source"] == attribution.REPORTED
    assert attribution.model_names(attribution.from_result(result)) == [
        "claude-sonnet-4-5-20250929"]


def test_claude_cli_records_every_model_the_session_used():
    """A session that switched models keys more than one — record all of them."""
    from graphify.llm import _call_claude_cli

    envelope = {
        "result": json.dumps({"nodes": [], "edges": [], "hyperedges": []}),
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "modelUsage": {"claude-opus-4-1": {}, "claude-sonnet-4-5": {}},
        "stop_reason": "end_turn",
    }
    result = _run_claude_cli_with(envelope, _call_claude_cli)

    assert attribution.model_names(attribution.from_result(result)) == [
        "claude-opus-4-1", "claude-sonnet-4-5"]
    assert attribution.is_mixed(attribution.from_result(result))


def test_claude_cli_without_model_usage_is_not_passed_off_as_confirmed():
    """The `claude-code-plan` placeholder is not something a provider said."""
    from graphify.llm import _call_claude_cli

    envelope = {
        "result": json.dumps({"nodes": [], "edges": [], "hyperedges": []}),
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
    }
    result = _run_claude_cli_with(envelope, _call_claude_cli)

    assert result["model"] == "claude-code-plan"
    assert result["model_source"] == attribution.REQUESTED


def _run_claude_cli_with(envelope, fn):
    """Drive `_call_claude_cli` against a canned Claude Code envelope."""
    import subprocess
    from unittest import mock

    completed = subprocess.CompletedProcess(
        args=["claude"], returncode=0, stdout=json.dumps(envelope), stderr=""
    )
    with mock.patch("subprocess.run", return_value=completed), \
            mock.patch("shutil.which", return_value="/usr/bin/claude"):
        return fn("some prompt", max_tokens=1024)


# ── retries and resumed runs keep their attribution ──────────────────────────

def test_merge_across_results_is_order_stable():
    a = attribution.attach({}, "m1", "m1")
    b = attribution.attach({}, "m2", "m2")
    assert attribution.model_names(attribution.merge(a, b)) == ["m1", "m2"]
    assert attribution.model_names(attribution.merge(b, a)) == ["m2", "m1"]


def test_merge_accepts_results_and_record_lists_together():
    """An incremental rebuild unions the prior graph's models with this run's."""
    result = attribution.attach({}, "m1", "m1")
    prior = [{"model": "m0", "source": attribution.REPORTED}]
    assert attribution.model_names(attribution.merge(prior, result)) == ["m0", "m1"]


def test_merge_ignores_junk_sources():
    assert attribution.merge(None, 5, "x", {"nope": 1}) == []


def test_bare_strings_are_treated_as_unconfirmed():
    """A bare model name carries no evidence a provider confirmed it."""
    assert attribution.normalize(["gpt-5"]) == [
        {"model": "gpt-5", "source": attribution.REQUESTED}]


def test_unknown_source_value_degrades_to_requested():
    assert attribution.record("gpt-5", "invented")["source"] == attribution.REQUESTED


@pytest.mark.parametrize("bad", [None, 5, "str", object()])
def test_attach_is_total(bad):
    """attach must never raise, whatever it is handed."""
    attribution.attach(bad, "m")
    attribution.attach({}, bad)
    attribution.attach({}, "m", bad)


# ── an AST-only rebuild must not erase the semantic model ────────────────────

def test_reexport_carries_attribution_forward():
    """`update`/`watch`/`cluster-only` rebuild the AST side and reuse the
    existing graph's semantic nodes, so they inherit its attribution. Losing it
    on every file save would make the field worthless."""
    G = make_graph()
    communities = cluster(G)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        to_json(G, communities, str(out), models=[
            {"model": "gemini-2.5-pro", "source": attribution.REPORTED}])

        # A later rebuild reads the prior graph and passes it back through.
        carried = attribution.from_artifact(json.loads(out.read_text()))
        to_json(G, communities, str(out), force=True, models=carried)

        assert attribution.model_names(
            attribution.from_artifact(json.loads(out.read_text()))
        ) == ["gemini-2.5-pro"]


def test_rebuild_of_an_old_graph_adds_no_model_key():
    """An old graph re-exported stays free of a models field, so the rebuild is
    byte-stable and no spurious rewrite is triggered."""
    G = make_graph()
    communities = cluster(G)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "graph.json"
        to_json(G, communities, str(out))
        carried = attribution.from_artifact(json.loads(out.read_text()))
        assert carried == []
        first = out.read_text()
        to_json(G, communities, str(out), force=True, models=carried)
        assert out.read_text() == first
