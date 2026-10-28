"""AI Transport Planning Assistant. Every LLM call is replaced by a scripted fake:
no API key or network is needed. The live model is scored by eval_assistant.py."""

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.assistant as assistant
from src.assistant import (
    PROPOSAL_SCHEMA,
    REQUEST_TYPES,
    AssistantError,
    AssistantUnavailable,
    assist,
    build_context,
    check_grounding,
    provider_from_config,
    validate_proposal,
)
from src.model import Constraints
from src.planner import optimise
from tests.assistant_eval_cases import CASES, llm

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeProvider:
    """Returns scripted outputs in order (or raises), and records every call."""

    label = "Fake provider"

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete_json(self, system, user, schema):
        self.calls.append((system, user, schema))
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return json.loads(json.dumps(out))  # same shape as a real JSON round trip


@pytest.fixture(scope="module")
def fixed_run(real_dataset):
    return optimise(real_dataset, "fixed", Constraints(), 20)


@pytest.fixture(scope="module")
def context(real_dataset, fixed_run):
    runs = {"fixed": {"solution": fixed_run.solution, "baseline": fixed_run.baseline, "current_inputs": True}}
    return build_context(real_dataset, Constraints(), runs, "fixed", fixed_run.solution)


def facts(context):
    return next(r for r in context["results"] if r["cost_model"] == "fixed")


def constraints_from(case):
    rules = case.get("rules") or {}
    return Constraints(list(rules.get("must_share", [])), list(rules.get("cannot_share", [])),
                       list(rules.get("solo", [])))


# --------------------------------------------------------------------------- #
# Evaluation set, offline: scripted model outputs through the real validation
# --------------------------------------------------------------------------- #


def test_eval_set_size_and_coverage():
    assert 20 <= len(CASES) <= 30
    assert {c["expected"] for c in CASES} == set(REQUEST_TYPES)
    assert sum(c["approve"] for c in CASES) >= 5 and sum(c.get("error") is not None for c in CASES) >= 10


@pytest.mark.parametrize("case", CASES, ids=[c["request"][:40] for c in CASES])
def test_eval_case(case, real_dataset, context):
    raw = dict(case["llm"])
    raw["answer"] = raw["answer"].format(**facts(context))
    provider = FakeProvider(raw)
    p = assist(case["request"], provider, real_dataset, constraints_from(case), context, "fixed")

    assert p.needs_approval == case["approve"], (p.errors, p.notes)
    if case.get("error"):
        assert any(case["error"] in e for e in p.errors), p.errors
    else:
        assert not p.errors, p.errors
    if case.get("note"):
        assert any(case["note"] in n for n in p.notes), p.notes
    if not case["approve"]:
        assert not p.attendance or p.errors  # nothing executable leaks through
    # Only the validated, allowlisted types ever come back.
    assert p.request_type in REQUEST_TYPES


def test_eval_summary_offline(real_dataset, context, capsys):
    """Measured guardrail result over the whole set (printed with -s)."""
    passed = 0
    for case in CASES:
        raw = dict(case["llm"])
        raw["answer"] = raw["answer"].format(**facts(context))
        p = assist(case["request"], FakeProvider(raw), real_dataset, constraints_from(case), context, "fixed")
        ok = p.needs_approval == case["approve"] and (
            any(case["error"] in e for e in p.errors) if case.get("error") else not p.errors)
        passed += ok
    print(f"\nAssistant guardrail eval (scripted outputs): {passed}/{len(CASES)} cases as expected")
    assert passed == len(CASES)


# --------------------------------------------------------------------------- #
# Specific guarantees
# --------------------------------------------------------------------------- #


def test_validated_change_carries_exact_values(real_dataset):
    p = validate_proposal(llm("attendance_change", att=[("np 11", 15)]), "r", real_dataset, Constraints(), "distance")
    assert p.needs_approval and p.attendance == {"NP11": 15} and p.current_attendance == {"NP11": 18}
    assert p.cost_model == "distance"  # selected model when the request names none


def test_proposal_never_mutates_inputs(real_dataset):
    before = real_dataset.schools.copy()
    c = Constraints()
    assist("NP11 now brings 15", FakeProvider(llm("attendance_change", att=[("NP11", 15)])), real_dataset, c,
           {}, "fixed")
    assert real_dataset.schools.equals(before) and c == Constraints()


@pytest.mark.parametrize("bad", [
    "not a dict", {"request_type": "explain"}, {**llm("explain", answer="x"), "answer": 5},
    {**llm("delete_everything"), "request_type": "delete_everything"},
    {**llm("attendance_change"), "attendance_changes": [["NP11", 15]]},
    {**llm("sharing_rule"), "sharing_rules": [{"rule": "teleport", "schools": ["NP11"]}]},
])
def test_malformed_model_output_is_rejected(bad, real_dataset):
    p = validate_proposal(bad, "r", real_dataset, Constraints(), "fixed")
    assert not p.needs_approval and p.errors


@pytest.mark.parametrize("exc", [AssistantError("No connection to the language model."), RuntimeError("boom")])
def test_provider_failures_are_reported_not_raised(exc, real_dataset):
    p = assist("Find the cheapest plan", FakeProvider(exc), real_dataset, Constraints(), {}, "fixed")
    assert not p.needs_approval and p.errors


def test_empty_and_overlong_requests_do_not_call_the_model(real_dataset):
    provider = FakeProvider(llm("run_optimiser", model="fixed"))
    assert assist("   ", provider, real_dataset, Constraints(), {}, "fixed").request_type == "clarify"
    assert assist("x" * 1001, provider, real_dataset, Constraints(), {}, "fixed").errors
    assert provider.calls == []


def test_context_has_no_coordinates_and_uses_structured_schema(real_dataset, context):
    provider = FakeProvider(llm("clarify", q="Which school?"))
    assist("help", provider, real_dataset, Constraints(), context, "fixed")
    system, user, schema = provider.calls[0]
    assert schema is PROPOSAL_SCHEMA and "<request>\nhelp\n</request>" in user
    assert "latitude" not in user and "longitude" not in user and "51.4" not in user
    assert all(set(s) <= {"school_id", "attendees", "bus", "bus_size"} for s in context["schools"])


def test_schema_is_strict():
    assert PROPOSAL_SCHEMA["additionalProperties"] is False
    assert set(PROPOSAL_SCHEMA["required"]) == set(PROPOSAL_SCHEMA["properties"])
    assert PROPOSAL_SCHEMA["properties"]["request_type"]["enum"] == list(REQUEST_TYPES)


def test_grounding_allows_context_figures_and_rejects_invented_ones(context):
    f = facts(context)
    ok = f"It costs {f['total_cost']:,.2f} cost units with {f['buses']} buses, saving {f['saving_percent']}%."
    assert check_grounding(ok, context) == []
    assert check_grounding("It costs 1,234.56 cost units.", context)


def test_grounding_allows_the_school_count(context):
    # Regression: "How many schools are there?" was withheld because 60 was not in the context.
    assert context["school_count"] == len(context["schools"]) == 60
    assert check_grounding("There are 60 schools.", context) == []


def test_grounding_rejects_optimal_claims_without_proof(context):
    unproven = {"results": [{"cost_model": "distance", "proven_optimal": False}]}
    assert check_grounding("This plan is optimal.", unproven)
    assert check_grounding("This plan is not proven optimal.", unproven) == []
    if facts(context)["proven_optimal"]:
        assert check_grounding("The fixed-cost plan is proven optimal.", context) == []


def test_answers_are_rendered_without_links_images_or_html(real_dataset):
    raw = llm("explain", answer='See ![x](https://evil.example/p?d=1) [here](https://evil.example) '
                                 'https://evil.example <img src="https://evil.example/a">done')
    p = validate_proposal(raw, "r", real_dataset, Constraints(), "fixed", {})
    assert "evil.example" not in p.answer and "<img" not in p.answer and p.answer.endswith("done")


@pytest.fixture
def no_llm_env(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CAMPUS_CRUISE_LLM_MODEL", "CAMPUS_CRUISE_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_missing_key_means_unavailable(no_llm_env):
    with pytest.raises(AssistantUnavailable, match="OPENAI_API_KEY"):
        provider_from_config({})


def test_key_from_secrets_and_model_override(no_llm_env):
    no_llm_env.setenv("CAMPUS_CRUISE_LLM_MODEL", "claude-sonnet-5-5")
    p = provider_from_config({"ANTHROPIC_API_KEY": "sk-test-not-real"})
    assert isinstance(p, assistant.AnthropicProvider)
    assert p.model == "claude-sonnet-5-5" and "sk-test" not in p.label


def test_openai_key_selects_openai(no_llm_env):
    p = provider_from_config({"OPENAI_API_KEY": "sk-proj-test-not-real"})
    assert isinstance(p, assistant.OpenAIProvider) and p.model == "gpt-4.1-mini"
    assert "sk-proj" not in p.label and p.vendor == "OpenAI"


def test_provider_choice_and_misplaced_key(no_llm_env):
    both = {"OPENAI_API_KEY": "sk-proj-x", "ANTHROPIC_API_KEY": "sk-ant-x"}
    assert isinstance(provider_from_config(both), assistant.OpenAIProvider)
    assert isinstance(provider_from_config({**both, "CAMPUS_CRUISE_LLM_PROVIDER": "anthropic"}),
                      assistant.AnthropicProvider)
    with pytest.raises(AssistantUnavailable, match="OpenAI key"):
        provider_from_config({"ANTHROPIC_API_KEY": "sk-proj-x"})
    with pytest.raises(AssistantUnavailable, match="must be"):
        provider_from_config({**both, "CAMPUS_CRUISE_LLM_PROVIDER": "other"})


class _Msg:
    def __init__(self, content, refusal=None):
        self.content, self.refusal = content, refusal


class _Choice:
    def __init__(self, content, finish="stop", refusal=None):
        self.message, self.finish_reason = _Msg(content, refusal), finish


class _Completion:
    def __init__(self, *a, **kw):
        self.choices = [_Choice(*a, **kw)]


def test_openai_request_shape(monkeypatch):
    p = assistant.OpenAIProvider("sk-proj-test-not-real")
    seen = {}

    def fake_create(**kw):
        seen.update(kw)
        return _Completion(json.dumps(llm("clarify", q="?")))

    monkeypatch.setattr(p._client.chat.completions, "create", fake_create)
    assert p.complete_json("sys", "user", PROPOSAL_SCHEMA)["request_type"] == "clarify"
    fmt = seen["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] is PROPOSAL_SCHEMA
    assert seen["messages"][0] == {"role": "system", "content": "sys"} and "tools" not in seen


@pytest.mark.parametrize("completion, message", [
    (_Completion("not json"), "invalid JSON"),
    (_Completion(None), "no answer"),
    (_Completion(None, refusal="I can't help with that."), "declined"),
    (_Completion("{}", finish="content_filter"), "declined"),
    (_Completion('{"a"', finish="length"), "cut off"),
])
def test_openai_bad_responses(monkeypatch, completion, message):
    p = assistant.OpenAIProvider("sk-proj-test-not-real")
    monkeypatch.setattr(p._client.chat.completions, "create", lambda **kw: completion)
    with pytest.raises(AssistantError, match=message):
        p.complete_json("s", "u", PROPOSAL_SCHEMA)


def test_openai_connection_and_auth_errors(monkeypatch):
    import httpx2
    import openai

    p = assistant.OpenAIProvider("sk-proj-test-not-real")
    req = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")

    def offline(**kw):
        raise openai.APIConnectionError(request=req)

    monkeypatch.setattr(p._client.chat.completions, "create", offline)
    with pytest.raises(AssistantError, match="No connection"):
        p.complete_json("s", "u", PROPOSAL_SCHEMA)


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Response:
    def __init__(self, text, stop="end_turn"):
        self.content, self.stop_reason = [_Block(text)], stop


@pytest.mark.parametrize("response, message", [
    (_Response("not json"), "invalid JSON"),
    (_Response("[1, 2]"), "not an object"),
    (_Response("{}", stop="refusal"), "declined"),
    (_Response("{", stop="max_tokens"), "cut off"),
])
def test_anthropic_provider_bad_responses(monkeypatch, response, message):
    p = assistant.AnthropicProvider("sk-test-not-real")
    monkeypatch.setattr(p._client.beta.messages, "create", lambda **kw: response)
    with pytest.raises(AssistantError, match=message):
        p.complete_json("s", "u", PROPOSAL_SCHEMA)


def test_anthropic_provider_request_shape(monkeypatch):
    p = assistant.AnthropicProvider("sk-test-not-real")
    seen = {}

    def fake_create(**kw):
        seen.update(kw)
        return _Response(json.dumps(llm("clarify", q="?")))

    monkeypatch.setattr(p._client.beta.messages, "create", fake_create)
    assert p.complete_json("sys", "user", PROPOSAL_SCHEMA)["request_type"] == "clarify"
    assert seen["model"] == "claude-opus-5-5"
    assert seen["output_config"]["format"] == {"type": "json_schema", "schema": PROPOSAL_SCHEMA}
    assert seen["fallbacks"] == "default" and "tools" not in seen  # no tools: the model cannot execute anything


def test_anthropic_provider_connection_error(monkeypatch):
    import anthropic
    import httpx2

    p = assistant.AnthropicProvider("sk-test-not-real")

    def boom(**kw):
        raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))

    monkeypatch.setattr(p._client.beta.messages, "create", boom)
    with pytest.raises(AssistantError, match="No connection"):
        p.complete_json("s", "u", PROPOSAL_SCHEMA)


# --------------------------------------------------------------------------- #
# Dashboard: request -> proposal -> approve / reject -> optimiser -> validator
# --------------------------------------------------------------------------- #


def app_with(monkeypatch, provider):
    monkeypatch.setattr(assistant, "provider_from_config", lambda secrets=None: provider)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.slider[0].set_value(5).run()  # fixed model, short limit
    return at


def ask(at, text):
    return at.chat_input(key="chat_input").set_value(text).run()


def test_app_approve_applies_change_and_reoptimises(monkeypatch):
    provider = FakeProvider(llm("attendance_change", "Set NP11 to 15.", att=[("NP11", 15)]))
    at = ask(app_with(monkeypatch, provider), "NP11 now brings 15 people.")
    assert not at.exception
    # Proposal shown; nothing applied, nothing run.
    assert at.session_state["assistant_proposal"].needs_approval
    assert at.session_state["overrides"] == {} and at.session_state["current"] is None
    assert any("NP11: 18 → 15" in str(d.value.values) for d in at.dataframe)

    at.button(key="chat_approve").click().run()
    assert not at.exception
    cur = at.session_state["current"]
    assert at.session_state["overrides"] == {"NP11": 15} and cur["overrides"] == {"NP11": 15}
    assert cur["model"] == "fixed" and cur["result"].solution.has_plan
    assert int(cur["dataset"].schools.set_index("school_id").loc["NP11", "attendees"]) == 15
    assert cur["result"].solution.total_attendees == 1241
    assert {m.label: m.value for m in at.metric}["Attendees"] == "1241"
    assert not cur["result"].solution.violations  # independent validator passed
    # The attendance editor now starts from the approved value.
    editor = at.session_state["editor_base"]
    assert editor == {"NP11": 15}
    assert at.session_state["assistant_log"][-1][2] == "approved and applied"


def test_app_reject_changes_nothing(monkeypatch):
    provider = FakeProvider(llm("sharing_rule", "Share.", rules=[("must_share", ("CF15", "CF38"))]))
    at = ask(app_with(monkeypatch, provider), "CF15 and CF38 must share a bus.")
    assert at.session_state["assistant_proposal"].needs_approval
    at.button(key="chat_reject").click().run()
    assert at.session_state["must_share"] == [] and at.session_state["current"] is None
    assert at.session_state["assistant_proposal"] is None
    assert any("Nothing was changed" in i.value for i in at.info)


def test_app_invalid_proposal_has_no_approve_button(monkeypatch):
    provider = FakeProvider(llm("attendance_change", att=[("NP11", 60)]))
    at = ask(app_with(monkeypatch, provider), "NP11 brings 60")
    assert not at.session_state["assistant_proposal"].needs_approval
    assert not any(b.key == "chat_approve" for b in at.button)
    assert any("largest bus" in e.value for e in at.error)


def test_app_without_key_still_works(monkeypatch):
    def unavailable(secrets=None):
        raise AssistantUnavailable("No API key.")

    monkeypatch.setattr(assistant, "provider_from_config", unavailable)
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert any("Assistant not configured" in i.value for i in at.info)
    assert at.chat_input(key="chat_input").disabled
    at.sidebar.slider[0].set_value(5).run()
    at.button(key="run_optimiser").click().run()
    assert at.session_state["current"]["result"].solution.has_plan
