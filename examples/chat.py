"""Interactive terminal chat for a running pico-sglang server.

The client uses pico-sglang's OpenAI-compatible API and keeps multiple conversations in
memory. Run ``python examples/chat.py`` and enter ``/help`` for available commands.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator


Message = dict[str, str]


class APIClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.timeout = timeout

    def model(self) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                models = json.load(response).get("data", [])
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot connect to {self.base_url}: {exc.reason}") from exc
        if not models:
            raise RuntimeError("the server returned no model from /v1/models")
        return str(models[0]["id"])

    def stream_chat(
        self,
        *,
        model      : str,
        messages   : list[Message],
        max_tokens : int,
        temperature: float,
        top_k      : int,
        top_p      : float,
        ignore_eos : bool,
    ) -> Iterator[str]:
        body = json.dumps({
            "model"      : model,
            "messages"   : messages,
            "max_tokens" : max_tokens,
            "temperature": temperature,
            "top_k"      : top_k,
            "top_p"      : top_p,
            "ignore_eos" : ignore_eos,
            "stream"     : True,
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Accept"      : "text/event-stream",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line.removeprefix("data:").strip()
                    if payload == "[DONE]":
                        break
                    choices = json.loads(payload).get("choices", [])
                    if choices and (piece := choices[0].get("delta", {}).get("content")):
                        yield piece
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace").strip()
            raise RuntimeError(f"server returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"connection failed: {exc.reason}") from exc


@dataclass
class Conversation:
    name    : str
    messages: list[Message] = field(default_factory=list)

    @property
    def turns(self) -> int:
        return sum(msg["role"] == "user" for msg in self.messages)

    @property
    def system_prompt(self) -> str | None:
        if self.messages and self.messages[0]["role"] == "system":
            return self.messages[0]["content"]
        return None

    def set_system_prompt(self, prompt: str | None) -> None:
        if self.system_prompt is not None:
            if prompt is None:
                self.messages.pop(0)
            else:
                self.messages[0]["content"] = prompt
        elif prompt is not None:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def clear(self) -> None:
        system_prompt = self.system_prompt
        self.messages.clear()
        self.set_system_prompt(system_prompt)


class Chat:
    COMMANDS = (
        "/help", "/new", "/list", "/switch", "/rename", "/delete",
        "/clear", "/system", "/history", "/exit",
    )

    def __init__(
        self,
        *,
        base_url    : str,
        model       : str | None,
        max_tokens  : int,
        temperature : float,
        top_k       : int,
        top_p       : float,
        ignore_eos  : bool,
        timeout     : float,
        system_prompt: str | None,
    ) -> None:
        self.client = APIClient(base_url, timeout)
        self.model = model or self.client.model()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.ignore_eos = ignore_eos

        first = Conversation("chat-1")
        first.set_system_prompt(system_prompt)
        self.conversations = {first.name: first}
        self.active_name = first.name
        self._next_id = 2

    @property
    def active(self) -> Conversation:
        return self.conversations[self.active_name]

    def _unique_name(self) -> str:
        while (name := f"chat-{self._next_id}") in self.conversations:
            self._next_id += 1
        self._next_id += 1
        return name

    def _resolve(self, value: str) -> Conversation:
        if value in self.conversations:
            return self.conversations[value]
        if value.isdigit():
            index = int(value) - 1
            conversations = list(self.conversations.values())
            if 0 <= index < len(conversations):
                return conversations[index]
        raise ValueError(f"unknown conversation: {value or '<empty>'}")

    def _help(self) -> None:
        print(
            "Commands:\n"
            "  /new [name]       create and enter a conversation\n"
            "  /list             list conversations\n"
            "  /switch <name|#>  switch by name or list number\n"
            "  /rename <name>    rename the current conversation\n"
            "  /delete [name|#]  delete a conversation\n"
            "  /clear            clear the current conversation\n"
            "  /system [text]    show/set its system prompt; 'off' removes it\n"
            "  /history          print the current conversation\n"
            "  /exit             exit"
        )

    def _list(self) -> None:
        for index, conversation in enumerate(self.conversations.values(), 1):
            marker = "*" if conversation.name == self.active_name else " "
            print(f" {marker} {index:>2}. {conversation.name} ({conversation.turns} turns)")

    def _history(self) -> None:
        if not self.active.messages:
            print("This conversation is empty.")
        for message in self.active.messages:
            print(f"\n{message['role']} › {message['content']}")

    def _command(self, value: str) -> bool:
        command, _, argument = value.partition(" ")
        argument = argument.strip()
        if command in ("/exit", "/quit"):
            return False
        if command == "/help":
            self._help()
        elif command == "/list":
            self._list()
        elif command == "/new":
            name = argument or self._unique_name()
            if name in self.conversations:
                raise ValueError(f"conversation already exists: {name}")
            conversation = Conversation(name)
            conversation.set_system_prompt(self.active.system_prompt)
            self.conversations[name] = conversation
            self.active_name = name
            print(f"Switched to {name}.")
        elif command == "/switch":
            if not argument:
                raise ValueError("usage: /switch <name|#>")
            self.active_name = self._resolve(argument).name
            print(f"Switched to {self.active_name}.")
        elif command == "/rename":
            if not argument:
                raise ValueError("usage: /rename <name>")
            if argument in self.conversations:
                raise ValueError(f"conversation already exists: {argument}")
            conversation = self.conversations.pop(self.active_name)
            conversation.name = argument
            self.conversations[argument] = conversation
            self.active_name = argument
        elif command == "/delete":
            conversation = self._resolve(argument) if argument else self.active
            del self.conversations[conversation.name]
            if not self.conversations:
                replacement = Conversation(self._unique_name())
                self.conversations[replacement.name] = replacement
            if conversation.name == self.active_name:
                self.active_name = next(iter(self.conversations))
            print(f"Deleted {conversation.name}; active conversation is {self.active_name}.")
        elif command == "/clear":
            self.active.clear()
            print(f"Cleared {self.active_name}.")
        elif command == "/system":
            if not argument:
                print(self.active.system_prompt or "No system prompt.")
            else:
                remove = argument.lower() == "off"
                self.active.set_system_prompt(None if remove else argument)
                print("System prompt removed." if remove else "System prompt updated.")
        elif command == "/history":
            self._history()
        else:
            raise ValueError(f"unknown command: {command}; use /help")
        return True

    def _chat(self, prompt: str) -> None:
        self.active.messages.append({"role": "user", "content": prompt})
        print("assistant › ", end="", flush=True)
        reply: list[str] = []
        try:
            stream = self.client.stream_chat(
                model=self.model,
                messages=self.active.messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                ignore_eos=self.ignore_eos,
            )
            for piece in stream:
                reply.append(piece)
                print(piece, end="", flush=True)
        except BaseException:
            self.active.messages.pop()
            print()
            raise
        print()
        self.active.messages.append({"role": "assistant", "content": "".join(reply)})

    def run(self) -> None:
        print("\npico-sglang chat")
        print(f"model: {self.model}")
        print("Use /help for commands; Ctrl-D exits.\n")
        self._configure_completion()

        while True:
            try:
                value = input(f"{self.active_name}  you › ").strip()
                if not value:
                    continue
                if value.startswith("/"):
                    if not self._command(value):
                        break
                else:
                    self._chat(value)
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\nCancelled. Use /exit or Ctrl-D to quit.")
            except (RuntimeError, ValueError) as exc:
                print(f"Error: {exc}")
            except Exception as exc:
                print(f"Request failed: {exc}")
        print("bye")

    def _configure_completion(self) -> None:
        try:
            import readline
        except ImportError:
            return

        def complete(text: str, state: int) -> str | None:
            options = [command for command in self.COMMANDS if command.startswith(text)]
            return options[state] if state < len(options) else None

        readline.set_completer(complete)
        readline.parse_and_bind("tab: complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--model", default=None, help="default: discover from /v1/models")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--system", default=None, help="initial system prompt")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        chat = Chat(
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            ignore_eos=args.ignore_eos,
            timeout=args.timeout,
            system_prompt=args.system,
        )
    except Exception as exc:
        print(f"Cannot connect to pico-sglang: {exc}", file=sys.stderr)
        return 1
    chat.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
