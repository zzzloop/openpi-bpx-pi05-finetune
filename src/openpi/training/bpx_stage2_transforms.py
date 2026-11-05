"""Extra token views used only by explicitly selected Stage-2 modes."""

from dataclasses import dataclass

from openpi import transforms


@dataclass(frozen=True)
class Stage2Tokenize:
    base: transforms.TokenizePrompt
    mode: str

    def __call__(self, data):
        prompt = data["prompt"]
        if not isinstance(prompt, str):
            prompt = prompt.item()
        result = self.base(data)
        if self.mode == "policy_cfg":
            # Use exactly the same tokenizer and normalized proprioception, not the default task.
            result["null_prompt"], result["null_prompt_mask"] = self.base.tokenizer.tokenize("", result["state"])
        elif self.mode == "ccfa_ours":
            result["task_token_mask"] = self.base.tokenizer.task_token_mask(prompt, result["state"])
        return result


def extra_tokens(inputs, mode):
    return tuple(Stage2Tokenize(t, mode) if isinstance(t, transforms.TokenizePrompt) else t for t in inputs)
