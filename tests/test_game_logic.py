"""Tests for the game logic module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from telephone.game_logic import (
    AgentConfig,
    RelayStep,
    build_judge_prompt,
    build_prompt,
    compute_diff,
    export_chain,
    run_chain,
    run_judge,
)


def test_build_prompt_repeat():
    result = build_prompt("repeat", "Hello world")
    assert result == "Repeat this exactly to the next agent: Hello world"


def test_build_prompt_paraphrase():
    result = build_prompt("paraphrase", "Hello world")
    assert result == "Paraphrase this to the next agent: Hello world"


def test_build_prompt_exaggerate():
    result = build_prompt("exaggerate", "Hello world")
    assert result == "Exaggerate this to the next agent: Hello world"


def test_build_prompt_improv():
    result = build_prompt("improv", "Hello world")
    assert "yes and" in result
    assert "Hello world" in result


def test_build_prompt_custom():
    result = build_prompt("custom", "Hello world", "Translate to French")
    assert result == "Translate to French: Hello world"


def test_build_prompt_unknown_style_defaults_to_repeat():
    result = build_prompt("nonexistent", "Hello")
    assert "Repeat" in result


@patch("telephone.game_logic.AIClient")
def test_run_chain(mock_client_cls):
    mock_instance = MagicMock()
    mock_instance.generate_response.side_effect = ["Step 1 output", "Step 2 output"]
    mock_client_cls.return_value = mock_instance

    agents = [
        AgentConfig("ollama", "localhost", 11434, "", "llama3", "repeat"),
        AgentConfig("ollama", "localhost", 11434, "", "llama3", "paraphrase"),
    ]

    steps = list(run_chain("Hello", agents))
    assert len(steps) == 2
    assert steps[0].input_message == "Hello"
    assert steps[0].output_message == "Step 1 output"
    assert steps[1].input_message == "Step 1 output"
    assert steps[1].output_message == "Step 2 output"


def test_compute_diff_identical():
    diff = compute_diff("Hello world", "Hello world")
    assert all(tag == "equal" for tag, _ in diff)


def test_compute_diff_insertion():
    diff = compute_diff("Hello world", "Hello beautiful world")
    tags = [tag for tag, _ in diff]
    assert "insert" in tags
    words = [w for tag, w in diff if tag == "insert"]
    assert "beautiful" in words


def test_compute_diff_deletion():
    diff = compute_diff("Hello beautiful world", "Hello world")
    tags = [tag for tag, _ in diff]
    assert "delete" in tags


def test_compute_diff_replacement():
    diff = compute_diff("The cat sat", "The dog sat")
    tags = [tag for tag, _ in diff]
    assert "delete" in tags
    assert "insert" in tags


def test_export_chain():
    steps = [
        RelayStep(0, "llama3", "repeat", "Hello", "Hello there"),
        RelayStep(1, "llama3", "paraphrase", "Hello there", "Greetings"),
    ]
    text = export_chain("Hello", steps)
    assert "Original message: Hello" in text
    assert "Agent 1" in text
    assert "Agent 2" in text
    assert "Final message: Greetings" in text


def test_export_chain_empty():
    text = export_chain("Hello", [])
    assert "Original message: Hello" in text
    assert "Final message" not in text


def test_relay_step_to_dict():
    step = RelayStep(0, "llama3", "repeat", "in", "out")
    d = step.to_dict()
    assert d["agent_index"] == 0
    assert d["model"] == "llama3"
    assert d["input_message"] == "in"
    assert d["output_message"] == "out"


def test_export_chain_with_judge():
    steps = [
        RelayStep(0, "llama3", "repeat", "Hello", "Hello there"),
    ]
    text = export_chain("Hello", steps, judge_analysis="This is the analysis.")
    assert "JUDGE ANALYSIS" in text
    assert "This is the analysis." in text


def test_build_judge_prompt():
    steps = [
        RelayStep(0, "llama3", "repeat", "Hello", "Hello there"),
        RelayStep(1, "llama3", "paraphrase", "Hello there", "Greetings"),
    ]
    prompt = build_judge_prompt("Hello", steps)
    assert "ORIGINAL MESSAGE:" in prompt
    assert "Hello" in prompt
    assert "AGENT 1" in prompt
    assert "AGENT 2" in prompt
    assert "FINAL MESSAGE:" in prompt
    assert "500-word analysis" in prompt


@patch("telephone.game_logic.AIClient")
def test_run_judge(mock_client_cls):
    mock_instance = MagicMock()
    mock_instance.generate_response.return_value = "Judge analysis result"
    mock_client_cls.return_value = mock_instance

    steps = [RelayStep(0, "llama3", "repeat", "Hello", "Hello there")]
    judge = AgentConfig("ollama", "localhost", 11434, "", "llama3", "judge")

    result = run_judge("Hello", steps, judge)
    assert result == "Judge analysis result"
    mock_instance.generate_response.assert_called_once()
