
import json
import os
import subprocess
import psutil
from groq import Groq


# ==========================================
# 1. GROQ CONFIGURATION
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "openai/gpt-oss-120b"

client = Groq(api_key=GROQ_API_KEY)


# ==========================================
# 2. DEFINE AGENT TOOLS (Python Functions)
# ==========================================

def get_system_metrics() -> str:
    """Returns local host system resource utilization (CPU, RAM, Disk)."""

    metrics = {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }

    return json.dumps(metrics)


def ping_host(hostname: str) -> str:
    """
    Pings a network host to evaluate connectivity.

    Args:
        hostname: Domain or IP to test.
    """

    try:

        res = subprocess.check_output(
            ["ping", "-n", "2", hostname],
            stderr=subprocess.STDOUT,
            text=True
        )

        return json.dumps({
            "status": "success",
            "raw_output": res.strip()
        })

    except Exception as err:

        return json.dumps({
            "status": "error",
            "message": str(err)
        })


# ==========================================
# 3. MAP FUNCTION NAMES TO PYTHON FUNCTIONS
# ==========================================

SYSTEM_TOOLS_MAP = {
    "get_system_metrics": get_system_metrics
}

NETWORK_TOOLS_MAP = {
    "ping_host": ping_host
}


# ==========================================
# 4. DEFINE JSON TOOL SCHEMAS
# ==========================================

SYSTEM_TOOLS_SCHEMA = [

    {
        "type": "function",

        "function": {
            "name": "get_system_metrics",

            "description":
                "Get current CPU, RAM, and Disk metrics of the machine.",

            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

]


NETWORK_TOOLS_SCHEMA = [

    {
        "type": "function",

        "function": {
            "name": "ping_host",

            "description":
                "Ping a specified hostname or IP address to verify connectivity.",

            "parameters": {

                "type": "object",

                "properties": {

                    "hostname": {
                        "type": "string",
                        "description":
                            "Target hostname or IP address."
                    }

                },

                "required": ["hostname"]
            }
        }
    }

]


# ==========================================
# 5. GENERIC AGENT RUNNER
# ==========================================

def run_agent(
    agent_name: str,
    system_prompt: str,
    user_query: str,
    tools_schema: list,
    tools_map: dict
):

    print(f"\n--- [{agent_name}] Activated ---")


    messages = [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "user",
            "content": user_query
        }

    ]


    # ======================================
    # AGENT LOOP
    # ======================================

    while True:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=messages,

            tools=tools_schema,

            tool_choice="auto"
        )


        assistant_message = response.choices[0].message


        # Add assistant message to conversation
        messages.append(assistant_message)


        # ==================================
        # NO TOOL CALL
        # ==================================

        if not assistant_message.tool_calls:

            print(f"\n[{agent_name} Response]:")

            print(
                assistant_message.content
            )

            return


        # ==================================
        # TOOL CALL
        # ==================================

        for call in assistant_message.tool_calls:

            fn_name = call.function.name

            fn_args = json.loads(
                call.function.arguments
            )


            print(
                f"\n  └─ Executing Tool: `{fn_name}`"
            )

            print(
                f"     Parameters: {fn_args}"
            )


            # Find Python function

            if fn_name in tools_map:

                try:

                    result = tools_map[fn_name](
                        **fn_args
                    )

                except Exception as err:

                    result = json.dumps({
                        "status": "error",
                        "message": str(err)
                    })

            else:

                result = json.dumps({
                    "status": "error",
                    "message": "Unknown tool"
                })


            print(
                f"     Result: {result}"
            )


            # ==================================
            # SEND TOOL RESULT BACK TO GROQ
            # ==================================

            messages.append({

                "role": "tool",

                "tool_call_id": call.id,

                "name": fn_name,

                "content": result

            })


        # Loop continues.
        # Groq now receives the tool result
        # and can produce the final answer.


# ==========================================
# 6. ORCHESTRATOR ROUTER
# ==========================================

def orchestrate_query(user_query: str):

    print(
        "\n=========================================="
    )

    print(
        f'USER QUERY: "{user_query}"'
    )

    print(
        "=========================================="
    )


    router_prompt = f"""

You are a query router.

Analyze the user prompt and respond with ONLY ONE word.

SYSTEM
if the query asks about:
- local CPU
- RAM
- memory
- disk
- system hardware
- computer resource usage

NETWORK
if the query asks about:
- pinging
- network latency
- internet connectivity
- network connection
- checking whether a host is reachable

UNKNOWN
if it fits neither.

Query:
{user_query}

"""


    # ======================================
    # GROQ ROUTER CALL
    # ======================================

    route_res = client.chat.completions.create(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": router_prompt
            }
        ]
    )


    decision = (
        route_res
        .choices[0]
        .message
        .content
        .strip()
        .upper()
    )


    print(
        f"\n[Router Decision]: {decision}"
    )


    # ======================================
    # SYSTEM AGENT
    # ======================================

    if "SYSTEM" in decision:

        run_agent(

            agent_name="System Health Agent",

            system_prompt="""
You are a system administration agent.

Use the get_system_metrics tool when the user
asks about CPU, RAM, memory, disk, or system
resource usage.

After receiving the tool result, clearly
explain the system status to the user.
""",

            user_query=user_query,

            tools_schema=SYSTEM_TOOLS_SCHEMA,

            tools_map=SYSTEM_TOOLS_MAP

        )


    # ======================================
    # NETWORK AGENT
    # ======================================

    elif "NETWORK" in decision:

        run_agent(

            agent_name="Network Agent",

            system_prompt="""
You are a network diagnostic agent.

Use the ping_host tool when the user asks
to ping a hostname or IP address or check
network connectivity.

After receiving the tool result, clearly
explain the connectivity status.
""",

            user_query=user_query,

            tools_schema=NETWORK_TOOLS_SCHEMA,

            tools_map=NETWORK_TOOLS_MAP

        )


    # ======================================
    # UNKNOWN
    # ======================================

    else:

        print(
            "\n[Router]: Request does not match "
            "active agent domains."
        )


# ==========================================
# 7. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print(
        "=========================================="
    )

    print(
        "      GROQ MULTI-AGENT SYSTEM"
    )

    print(
        "=========================================="
    )

    print(
        "Type 'exit' to quit."
    )


    while True:

        user_query = input(
            "\nEnter your query: "
        ).strip()


        if user_query.lower() == "exit":

            print(
                "\nAgent stopped."
            )

            break


        if not user_query:

            continue


        orchestrate_query(
            user_query
        )
