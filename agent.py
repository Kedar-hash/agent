import json
import os
import subprocess
import psutil
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "llama-3.3-70b-versatile"


def echo_fun(content: str):
    return f"Echo: {content}"


def function1(content: str):
    return content.upper()


def function2():
    return "Function 2 executed"


def function3():
    return "Function 3 executed"


TOOLS = [echo_fun, function1, function2, function3]


SYSTEM_PROMPT = """
You are a model which likes to eat kombadi vade and chicken biryani as well.
Your job is to read the tools and print the message.
Use echo_fun to echo text.
Use function1 to convert text into uppercase.
"""


def agent_run():

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    print("Local Qwen3 Echo Agent")
    print("Type 'exit' to quit the agent.")

    while True:

        prompt = input(
            "\nEnter file details: "
        ).strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        while True:

            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )

            messages.append(response.message)

            if not response.message.tool_calls:

                print(
                    "\nAgent:",
                    response.message.content
                )

                break

            for call in response.message.tool_calls:

                name = call.function.name
                args = call.function.arguments

                tool = next(
                    (
                        t
                        for t in TOOLS
                        if t.__name__ == name
                    ),
                    None
                )

                if tool is None:

                    result = "Error: Unknown tool."

                else:

                    try:
                        result = tool(**args)

                    except Exception as exc:
                        result = f"Error: {exc}"

                print(f"\n[Tool: {name}]")
                print("Result:", result)

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": name,
                        "content": str(result),
                    }
                )


if __name__ == "__main__":
    agent_run()