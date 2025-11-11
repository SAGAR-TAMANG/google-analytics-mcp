# main.py
import os
import argparse
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -------- Pretty terminal helpers (ANSI) --------
RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
ITALIC = "\x1b[3m"
FG_GREEN = "\x1b[32m"
FG_CYAN = "\x1b[36m"
FG_YELLOW = "\x1b[33m"
FG_MAGENTA = "\x1b[35m"
FG_RED = "\x1b[31m"
FG_BLUE = "\x1b[34m"
GRAY = "\x1b[90m"

def bar(title, color=FG_CYAN):
    print(f"\n{color}{BOLD}── {title} ─────────────────────────────────────────{RESET}")

def kv(label, value, color_label=GRAY, color_value=RESET, indent=2):
    pad = " " * indent
    print(f"{pad}{color_label}{label}:{RESET} {color_value}{value}{RESET}")

def pretty_json(obj, indent=2):
    return json.dumps(obj, ensure_ascii=False, indent=indent)

# ------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simple CLI chat with OpenAI (Responses API) + live event trace.")
    parser.add_argument("-m", "--message", default=None, help="Initial user message")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name (e.g., gpt-5, gpt-5-mini, gpt-4o-mini)")
    parser.add_argument("--no-tools", action="store_true", help="Disable MCP/tool usage")
    parser.add_argument("--trace", action="store_true", help="Show detailed tool/MCP event logs")
    args = parser.parse_args()

    client = OpenAI()

    # Conversation history (Responses API is stateless unless you include prior turns).
    history = [
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You are a helpful assistant in a CLI. "
                        "When appropriate, include a very brief section titled "
                        "'Reasoning (brief):' with 1–2 short sentences that summarize your plan—"
                        "do not reveal chain-of-thought or step-by-step reasoning."
                    ),
                }
            ],
        }
    ]

    tools = None if args.no_tools else [
        {
            "type": "mcp",
            "server_label": "google-analytics-mcp",
            "server_url": "https://tom-nominated-beneficial-joining.trycloudflare.com/mcp/",
            "require_approval": "never",
        },
    ]

    def ask(model: str, user_text: str) -> str:
        # Append user turn
        history.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}],
            }
        )

        bar("Assistant", FG_GREEN)
        print(f"{FG_GREEN}{BOLD}Assistant:{RESET} ", end="", flush=True)

        # Stream the response so we can print events as they happen.
        try:
            with client.responses.stream(
                model=model,
                input=history,
                tools=tools or None,
            ) as stream:
                # Handle incoming semantic events
                for event in stream:
                    t = getattr(event, "type", None)

                    # 1) Main streamed text
                    if t == "response.output_text.delta":
                        # Token deltas
                        sys.stdout.write(event.delta)
                        sys.stdout.flush()

                    # 2) Completed text chunk
                    elif t == "response.output_text.done":
                        print("", end="")  # keep on same line

                    # 3) Tool/MCP calls (traceable events)
                    elif args.trace and t == "response.tool_call.begin":
                        bar("Tool Call Begin", FG_YELLOW)
                        kv("tool", event.tool.name, FG_YELLOW)
                        if getattr(event, "call_id", None):
                            kv("call_id", event.call_id, FG_YELLOW)
                        if getattr(event, "args", None):
                            print(pretty_json(event.args))

                    elif args.trace and t == "response.tool_call.delta":
                        # Streaming tool arguments/stdin etc.
                        if getattr(event, "args_delta", None):
                            kv("args_delta", "", FG_YELLOW)
                            print(pretty_json(event.args_delta))

                    elif args.trace and t == "response.tool_call.output":
                        bar("Tool Output", FG_BLUE)
                        # Some tools stream text; some return structured output
                        if getattr(event, "output_text", None):
                            print(event.output_text, end="" if event.is_streaming else "\n")
                        if getattr(event, "output_image", None):
                            kv("output_image", "[binary image]", FG_BLUE)

                    elif args.trace and t == "response.tool_call.completed":
                        bar("Tool Call Completed", FG_YELLOW)
                        if getattr(event, "result", None):
                            print(pretty_json(event.result))

                    # 4) Errors
                    elif t == "response.error":
                        bar("Error", FG_RED)
                        print(pretty_json({"error": event.error}))
                        break

                # Get the final Response object (includes full message, tool results, etc.)
                resp = stream.get_final_response()

        except KeyboardInterrupt:
            raise
        except Exception as e:
            bar("Error", FG_RED)
            print(f"{FG_RED}{e}{RESET}")
            return ""

        # Pull assistant’s plain text for history + return
        reply = resp.output_text or ""
        history.append(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": reply}],
            }
        )

        # Optionally show the model-provided short rationale, if present.
        # We’ll scan the final text for the heading "Reasoning (brief):".
        # (This is *not* chain-of-thought—just a compact summary the model intentionally includes.)
        maybe = "\n" + reply
        marker = "\nReasoning (brief):"
        # if "Reasoning (brief):" in maybe:
        #     bar("Reasoning (brief)", FG_MAGENTA)
        #     # print everything beneath the marker until the next blank line
        #     lines = maybe.splitlines()
        #     start = next((i for i, s in enumerate(lines) if "Reasoning (brief):" in s), None)
        #     if start is not None:
        #         for j in range(start, len(lines)):
        #             if j > start and lines[j].strip() == "":
        #                 break
        #             print(lines[j])

        return reply

    try:
        # Optional first turn from CLI flag
        if args.message:
            bar("You", FG_CYAN)
            print(f"{BOLD}You:{RESET} {args.message}")
            ask(args.model, args.message)

        # Interactive loop
        while True:
            bar("You", FG_CYAN)
            user_text = input(f"{BOLD}You:{RESET} ").strip()
            if not user_text:
                continue
            ask(args.model, user_text)

    except KeyboardInterrupt:
        print(f"\n{DIM}Bye! 👋{RESET}")

if __name__ == "__main__":
    main()
