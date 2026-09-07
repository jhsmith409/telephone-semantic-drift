# Copyright (c) 2026 James H. Smith. MIT License.
"""Chain execution, prompt templates, and diff computation."""

from __future__ import annotations

import difflib
from collections.abc import Generator
from dataclasses import dataclass

from telephone.ai_client import AIClient

PROMPT_STYLES: dict[str, str] = {
    "repeat": "Repeat this exactly to the next agent: {message}",
    "paraphrase": "Paraphrase this to the next agent: {message}",
    "exaggerate": "Exaggerate this to the next agent: {message}",
    "improv": (
        'Restate this as part of an improv skit where you say "yes and..." '
        "retelling the original prompt and adding something to it: {message}"
    ),
    "custom": "{custom_prompt}: {message}",
}


@dataclass
class AgentConfig:
    service_type: str  # "ollama" or "openai"
    host: str
    port: int
    api_key: str
    model: str
    prompt_style: str  # key into PROMPT_STYLES
    custom_prompt: str = ""
    temperature: float | None = None
    system_prompt: str = ""
    seed: int | None = None


@dataclass
class RelayStep:
    agent_index: int
    model: str
    prompt_style: str
    input_message: str
    output_message: str

    def to_dict(self) -> dict:
        return {
            "agent_index": self.agent_index,
            "model": self.model,
            "prompt_style": self.prompt_style,
            "input_message": self.input_message,
            "output_message": self.output_message,
        }


def build_prompt(style: str, message: str, custom_prompt: str = "") -> str:
    """Build the user prompt for a given style and message."""
    template = PROMPT_STYLES.get(style, PROMPT_STYLES["repeat"])
    return template.format(message=message, custom_prompt=custom_prompt)


def run_chain(
    initial_message: str, agents: list[AgentConfig]
) -> Generator[RelayStep, None, None]:
    """Execute the telephone chain, yielding each step as it completes."""
    current_message = initial_message

    for i, agent in enumerate(agents):
        client = AIClient(
            service_type=agent.service_type,
            host=agent.host,
            port=agent.port,
            api_key=agent.api_key,
        )
        prompt = build_prompt(
            agent.prompt_style, current_message, agent.custom_prompt
        )
        output = client.generate_response(
            model=agent.model,
            system_prompt=agent.system_prompt,
            user_message=prompt,
            temperature=agent.temperature,
            seed=agent.seed,
        )

        step = RelayStep(
            agent_index=i,
            model=agent.model,
            prompt_style=agent.prompt_style,
            input_message=current_message,
            output_message=output,
        )
        yield step
        current_message = output


def compute_diff(original: str, final: str) -> list[tuple[str, str]]:
    """Compute a word-level diff between original and final text.

    Returns a list of (tag, word) tuples where tag is one of:
    'equal', 'insert', 'delete', 'replace'.
    """
    original_words = original.split()
    final_words = final.split()
    matcher = difflib.SequenceMatcher(None, original_words, final_words)

    result: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for w in original_words[i1:i2]:
                result.append(("equal", w))
        elif tag == "delete":
            for w in original_words[i1:i2]:
                result.append(("delete", w))
        elif tag == "insert":
            for w in final_words[j1:j2]:
                result.append(("insert", w))
        elif tag == "replace":
            for w in original_words[i1:i2]:
                result.append(("delete", w))
            for w in final_words[j1:j2]:
                result.append(("insert", w))
    return result


def build_judge_prompt(initial: str, steps: list[RelayStep]) -> str:
    """Build the prompt for the judge model."""
    parts = [
        "You are analyzing the results of an AI Telephone Game experiment. "
        "In this game, a message is passed through a chain of AI agents, "
        "each one transforming it before passing it on.\n\n"
        f"ORIGINAL MESSAGE:\n{initial}\n"
    ]
    for step in steps:
        parts.append(
            f"\nAGENT {step.agent_index + 1} ({step.model}, style: {step.prompt_style}):\n"
            f"Input: {step.input_message}\n"
            f"Output: {step.output_message}"
        )
    if steps:
        parts.append(f"\n\nFINAL MESSAGE:\n{steps[-1].output_message}")
    parts.append(
        "\n\nWrite a 500-word analysis of this experiment. Describe how the message "
        "changed at each step, what was preserved, what was lost or distorted, "
        "any interesting patterns, and what this reveals about how AI models "
        "process and transform information."
    )
    return "\n".join(parts)


def run_judge(
    initial: str, steps: list[RelayStep], judge: AgentConfig
) -> str:
    """Run the judge model to analyze the chain results."""
    client = AIClient(
        service_type=judge.service_type,
        host=judge.host,
        port=judge.port,
        api_key=judge.api_key,
    )
    prompt = build_judge_prompt(initial, steps)
    return client.generate_response(
        model=judge.model,
        system_prompt="You are an expert analyst of AI behavior and language transformation.",
        user_message=prompt,
        temperature=judge.temperature,
    )


def export_chain(initial: str, steps: list[RelayStep], judge_analysis: str = "") -> str:
    """Export the full chain as a readable text document."""
    lines = ["AI Telephone Game — Chain Export", "=" * 40, ""]
    lines.append(f"Original message: {initial}")
    lines.append("")

    for step in steps:
        lines.append(f"--- Agent {step.agent_index + 1} ({step.model}, {step.prompt_style}) ---")
        lines.append(f"Input:  {step.input_message}")
        lines.append(f"Output: {step.output_message}")
        lines.append("")

    if steps:
        lines.append("=" * 40)
        lines.append(f"Final message: {steps[-1].output_message}")

    if judge_analysis:
        lines.append("")
        lines.append("=" * 40)
        lines.append("JUDGE ANALYSIS")
        lines.append("=" * 40)
        lines.append(judge_analysis)

    return "\n".join(lines)
