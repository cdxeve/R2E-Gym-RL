#!/root/.venv/bin/python
"""
Description: Execute a bash command in the terminal, with Python version compatibility.

Parameters:
  --command (string, optional): The bash command to execute. For example: --command 'python my_script.py'. If not provided, will show help.
"""

import argparse
import subprocess
import sys

BLOCKED_BASH_COMMANDS = ["git", "ipython", "jupyter", "nohup"]


def run_command(cmd):
    # Use Python 3.6-compatible parameters (no capture_output or text)
    return subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Execute a bash command.")
    parser.add_argument(
        "command",
        type=str,
        help="The command (and optional arguments) to execute. For example: 'python my_script.py'",
    )
    args = parser.parse_args()

    # Check if the command is blocked
    first_token = args.command.strip().split()[0]
    if first_token in BLOCKED_BASH_COMMANDS:
        print(
            "Bash command '{}' is not allowed. "
            "Please use a different command or tool.".format(first_token)
        )
        sys.exit(1)

    result = run_command(args.command)

    if result.returncode != 0:
        print("Error executing command (exit code {}):\n".format(result.returncode))
        stdout_text = result.stdout.strip() if result.stdout else ""
        stderr_text = result.stderr.strip() if result.stderr else ""
        if stdout_text:
            print("[STDOUT]\n")
            print(stdout_text, "\n")
        if stderr_text:
            print("[STDERR]\n")
            print(stderr_text)
        if not stdout_text and not stderr_text:
            print("(no stdout or stderr output)")
        sys.exit(result.returncode)

    print(result.stdout.strip() if result.stdout else "")
    stderr_text = result.stderr.strip() if result.stderr else ""
    if stderr_text:
        print("\n[STDERR]\n")
        print(stderr_text)


if __name__ == "__main__":
    main()
