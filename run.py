"""CLI for Bring Books to Life.

    python run.py                 # chat with Sherlock Holmes (interactive)
    python run.py sherlock -v     # show when the character searches the book
    python run.py --demo          # a scripted 4-turn convo (tests retrieval + memory)
"""
import sys

from agent import BookChat
from personas import DEFAULT

DEMO = [
    "Who are you, and how would you describe yourself?",
    "What exactly is your method for cracking a case?",
    "And what do you make of your companion, Dr. Watson?",
    "Remind me — what was the very first thing I asked you?",   # <- tests memory
]


def main() -> None:
    args = sys.argv[1:]
    verbose = "-v" in args or "--verbose" in args
    demo = "--demo" in args
    positional = [a for a in args if not a.startswith("-")]
    persona = positional[0] if positional else DEFAULT

    chat = BookChat(persona)
    print(f"\n🎭  Bring Books to Life — you're speaking with {chat.persona.name}.\n")

    if demo:
        for msg in DEMO:
            print(f"You: {msg}")
            reply = chat.say(msg, verbose=verbose)
            print(f"\n{chat.persona.name}: {reply}\n{'-'*60}")
        return

    print("(type a message, Ctrl-C to leave)\n")
    while True:
        try:
            msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{chat.persona.name}: Farewell, my dear fellow.")
            break
        if not msg:
            continue
        reply = chat.say(msg, verbose=verbose)
        print(f"\n{chat.persona.name}: {reply}\n")


if __name__ == "__main__":
    main()
