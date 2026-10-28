"""Floating assistant chat (bottom-right popover). Every LLM call goes to a
scripted FakeProvider: no API key or network. The OR-Tools optimiser is real.

AppTest renders popover contents whether or not the popover is open in a
browser, so the chat widgets are reachable by key. Opening and closing the
popover is browser-only (``on_change="ignore"``) and never reruns the script.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.assistant as assistant
from src.assistant import AssistantError, AssistantUnavailable
from tests.assistant_eval_cases import llm
from tests.test_assistant import FakeProvider

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def app_with(monkeypatch, provider, run_first=False):
    monkeypatch.setattr(assistant, "provider_from_config", lambda secrets=None: provider)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.slider[0].set_value(5).run()  # fixed model, short limit
    if run_first:
        at.button(key="run_optimiser").click().run()
    return at


def send(at, text):
    return at.chat_input(key="chat_input").set_value(text).run()


def history(at):
    return at.session_state["chat_history"]


def test_floating_chat_has_one_typing_bar(monkeypatch):
    at = app_with(monkeypatch, FakeProvider(llm("clarify", q="Which school?")))
    assert not at.exception
    assert len(at.chat_input) == 1 and not at.chat_input(key="chat_input").disabled
    assert not any((b.key or "").startswith("chat_suggest_") for b in at.button)  # no suggestion buttons
    assert history(at) == []


def test_newest_answer_is_shown_first(monkeypatch):
    at = app_with(monkeypatch, FakeProvider(llm("clarify", q="Which school do you mean?")))
    send(at, "first question")
    send(at, "second question")
    users = [m.markdown[0].value for m in at.chat_message if m.name == "user"]
    assert users == ["second question", "first question"]
    # Each request is followed by its own reply.
    roles = [m.name for m in at.chat_message]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_read_only_question_uses_current_plan_and_changes_nothing(monkeypatch):
    provider = FakeProvider(llm("explain", "Explain plan.", "The current plan uses the fixed cost model."))
    at = app_with(monkeypatch, provider, run_first=True)
    cur, runs = at.session_state["current"], at.session_state["solver_runs"]
    send(at, "Explain the current plan.")
    assert not at.exception
    assert len(provider.calls) == 1
    sent = provider.calls[0][1]
    sol = cur["result"].solution
    assert sol.status_text() in sent                       # real solver status, not invented
    assert f"{sol.total_cost:.2f}".rstrip("0").rstrip(".") in sent  # real plan cost
    assert "51.4" not in sent and "-3.1" not in sent       # no coordinates leave the app
    assert at.session_state["current"] is cur and at.session_state["solver_runs"] == runs
    assert [m["role"] for m in history(at)] == ["user", "assistant"]
    assert history(at)[1]["proposal"].request_type == "explain"
    assert not any((b.key or "").endswith("_approve") for b in at.button)
    assert any("fixed cost model" in md.value for md in at.markdown)


def test_change_from_chat_needs_approval_then_runs_real_optimiser(monkeypatch):
    provider = FakeProvider(llm("attendance_change", "Set NP11 to 15.", att=[("NP11", 15)]))
    at = app_with(monkeypatch, provider, run_first=True)
    runs = at.session_state["solver_runs"]
    send(at, "NP11 now brings 15 people.")
    assert at.session_state["assistant_proposal"].needs_approval
    assert at.session_state["overrides"] == {}                     # nothing applied yet
    assert at.session_state["solver_runs"] == runs                 # nothing run yet
    assert at.button(key="chat_approve")

    at.button(key="chat_approve").click().run()
    assert not at.exception
    cur = at.session_state["current"]
    assert at.session_state["overrides"] == {"NP11": 15} and cur["overrides"] == {"NP11": 15}
    assert at.session_state["solver_runs"] == runs + 1 and cur["trigger"] == "assistant"
    assert cur["result"].solution.has_plan and not cur["result"].solution.violations
    assert history(at)[-1]["kind"] == "success"
    assert not any(b.key == "chat_approve" for b in at.button)    # decided: no second approval


def test_reject_from_chat_changes_nothing(monkeypatch):
    provider = FakeProvider(llm("sharing_rule", "Share.", rules=[("must_share", ("CF15", "CF38"))]))
    at = app_with(monkeypatch, provider)
    send(at, "CF15 and CF38 must share a bus.")
    at.button(key="chat_reject").click().run()
    assert at.session_state["must_share"] == [] and at.session_state["current"] is None
    assert at.session_state["solver_runs"] == 0
    assert "Nothing was changed" in history(at)[-1]["text"]


@pytest.mark.parametrize("output, message", [
    (AssistantError("No connection to the language model. Check the internet connection."), "No connection"),
    (AssistantError("The language model did not answer in time. Try again."), "did not answer in time"),
    ({"request_type": "delete_everything"}, "Not applied"),  # invalid model response
])
def test_failures_are_shown_in_chat_and_app_keeps_working(monkeypatch, output, message):
    at = app_with(monkeypatch, FakeProvider(output), run_first=True)
    cur = at.session_state["current"]
    send(at, "Find the cheapest valid transport plan.")
    assert not at.exception
    assert any(message in e.value for e in at.error)
    assert not any((b.key or "").endswith("_approve") for b in at.button)
    assert at.session_state["current"] is cur


def test_without_key_chat_is_disabled_and_app_works(monkeypatch):
    def unavailable(secrets=None):
        raise AssistantUnavailable("No API key.")

    monkeypatch.setattr(assistant, "provider_from_config", unavailable)
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    assert at.chat_input(key="chat_input").disabled
    assert any("Assistant not configured" in i.value for i in at.info)
    at.sidebar.slider[0].set_value(5).run()
    at.button(key="run_optimiser").click().run()
    assert at.session_state["current"]["result"].solution.has_plan


def test_chat_messages_never_rerun_optimiser(monkeypatch):
    provider = FakeProvider(llm("clarify", q="Which school do you mean?"))
    at = app_with(monkeypatch, provider, run_first=True)  # fixed: auto re-optimise is on
    cur, runs = at.session_state["current"], at.session_state["solver_runs"]
    for text in ("hello", "which bus?", "and now?"):
        send(at, text)
    at.button(key="chat_clear").click().run()
    assert len(provider.calls) == 3
    assert at.session_state["solver_runs"] == runs and at.session_state["current"] is cur
    assert history(at) == []


def test_clear_discards_pending_proposal(monkeypatch):
    provider = FakeProvider(llm("attendance_change", "Set NP11 to 15.", att=[("NP11", 15)]))
    at = app_with(monkeypatch, provider)
    send(at, "NP11 now brings 15 people.")
    at.button(key="chat_clear").click().run()
    assert at.session_state["assistant_proposal"] is None and at.session_state["overrides"] == {}
    assert at.session_state["assistant_log"][-1][2] == "discarded when the chat was cleared"


def test_chat_is_the_only_assistant_entry_point(monkeypatch):
    at = app_with(monkeypatch, FakeProvider(llm("clarify", q="Which school?")))
    assert "AI Assistant" not in [t.label for t in at.tabs]  # the separate tab was removed
    assert not any(b.key in ("assistant_ask", "assistant_approve") for b in at.button)
    assert any(e.label == "What is sent to the language model" for e in at.expander)
