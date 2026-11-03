import re
from typing import Dict
from r2egym.agenthub.action import Action
from r2egym.agenthub import CONTINUE_MSG


MAX_OBSERVATION_CHARS = 30000


class Observation:
    def __init__(self, bash_output, error_code, action: Action, num_lines: int=40,
                 max_chars: int = MAX_OBSERVATION_CHARS):
        self.bash_output = bash_output
        self.error_code = error_code
        self.action = action
        self.num_lines = num_lines
        self.max_chars = max_chars

    @staticmethod
    def _truncate_chars(text, max_chars):
        """Apply character-level truncation, keeping top and bottom portions."""
        if not text or len(text) <= max_chars:
            return text
        keep = max_chars // 2
        divider = "\n" + "-" * 50
        msg = (
            "\n<Observation truncated due to large output ({} chars total). "
            "Showing first and last {} chars.>".format(len(text), keep)
        )
        return text[:keep] + divider + msg + divider + "\n" + text[-keep:]

    def __str__(self):
        # empty or no function call
        if not self.action.function_name:
            return CONTINUE_MSG
        elif self.action.function_name == "finish" or self.action.function_name == "submit":
            return "<<< Finished >>>"
        else:
            if self.action.function_name == "execute_bash" or self.action.function_name == "bash":
                lines = self.bash_output.splitlines() if self.bash_output else []
                if len(lines) > 2 * self.num_lines:
                    top_lines = "\n".join(lines[:self.num_lines])
                    bottom_lines = "\n".join(lines[-self.num_lines:])
                    divider = "-"*50
                    truncated_output = (
                        f"{top_lines}\n"
                        f"{divider}\n"
                        f"<Observation truncated in middle for saving context>\n"
                        f"{divider}\n"
                        f"{bottom_lines}"
                    )
                else:
                    truncated_output = self.bash_output
                output = (
                    f"Exit code: {self.error_code}\n"
                    f"Execution output of [{self.action.function_name}]:\n"
                    f"{truncated_output}"
                )
            else:
                output = f"Execution output of [{self.action.function_name}]:\n{self.bash_output}"
            return self._truncate_chars(output, self.max_chars)
