"""
src/tokenizer/chat_template.py
Standardized Jinja2 chat template engine supporting ChatML, Llama-3, DeepSeek-R1 reasoning,
and Alpaca formats with assistant loss masking boundaries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

BUILTIN_TEMPLATES: Dict[str, str] = {
    "chatml": (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    ),
    "deepseek_r1": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{'<|im_start|>system\n' + message['content'] + '<|im_end|>\n'}}"
        "{% elif message['role'] == 'user' %}"
        "{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if message.get('thought') %}"
        "{{'<|im_start|>assistant\n<think>\n' + message['thought'] + '\n</think>\n' + message['content'] + '<|im_end|>\n'}}"
        "{% else %}"
        "{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n'}}"
        "{% endif %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n<think>\n' }}"
        "{% endif %}"
    ),
    "llama3": (
        "{{ '<|begin_of_text|>' }}"
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    ),
    "alpaca": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '### Instruction:\n' + message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '### Response:\n' + message['content'] + '\n\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '### Response:\n' }}"
        "{% endif %}"
    ),
}


class ChatTemplateRenderer:
    """Renders multi-turn chat messages using Jinja2 templates into text and prompt/completion pairs."""

    def __init__(self, template: str = "chatml") -> None:
        try:
            import jinja2
            self.jinja2 = jinja2
        except ImportError:
            self.jinja2 = None
            logger.warning("jinja2 package not installed; falling back to rule-based template formatting.")

        self.template_name = template if template in BUILTIN_TEMPLATES else "custom"
        self.raw_template = BUILTIN_TEMPLATES.get(template, template)
        self._compiled_template = None
        if self.jinja2:
            try:
                env = self.jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
                self._compiled_template = env.from_string(self.raw_template)
            except Exception as e:
                logger.warning("Could not compile jinja2 chat template: %s", e)

    def render(
        self,
        messages: List[Dict[str, Any]],
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        """Renders conversational message sequence into flat text with demarcation tokens."""
        if self._compiled_template:
            try:
                return self._compiled_template.render(
                    messages=messages,
                    add_generation_prompt=add_generation_prompt,
                    **kwargs,
                )
            except Exception as e:
                logger.warning("Error rendering jinja2 template: %s. Using fallback.", e)

        # Robust rule-based fallback (ChatML format)
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            thought = m.get("thought")
            if role == "assistant" and thought:
                out.append(f"<|im_start|>{role}\n<think>\n{thought}\n</think>\n{content}<|im_end|>\n")
            else:
                out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)

    def format_pair(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        chosen: str,
        rejected: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Formats pairwise alignment samples for DPO, SimPO, and ORPO.
        Returns dict with formatted 'prompt', 'chosen', and 'rejected'.
        """
        if isinstance(prompt, str):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = list(prompt)

        formatted_prompt = self.render(messages, add_generation_prompt=True)
        return {
            "prompt": formatted_prompt,
            "chosen": chosen.strip(),
            "rejected": rejected.strip(),
            "full_chosen": formatted_prompt + chosen.strip() + "<|im_end|>",
            "full_rejected": formatted_prompt + rejected.strip() + "<|im_end|>",
        }
