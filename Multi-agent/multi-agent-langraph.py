
import os
from typing import TypedDict

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# PDF imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# Email imports
import smtplib
from email.message import EmailMessage


# Use this project's .env values, even if PowerShell has an older key exported.
load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to the project's .env file or set it "
        "in the environment before running this script."
    )


# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=groq_api_key
)


# Shared state for all agents
class AgentState(TypedDict):
    question: str
    research: str
    technical: str
    report: str


# Coordinator node
def coordinator(state: AgentState):
    print("\n[Coordinator] Starting workflow")

    return {
        "research": "",
        "technical": "",
        "report": ""
    }


# Agent 1: Research Agent
def research_agent(state: AgentState):
    print("\n[Research Agent] Working...")

    prompt = f"""
    Analyze the following question as a Research Agent.

    Identify:
    - Important concepts
    - Requirements
    - Benefits
    - Challenges

    Question: {state['question']}
    """

    response = llm.invoke([
        SystemMessage(content="You are an IT Research Agent."),
        HumanMessage(content=prompt)
    ])

    return {"research": response.content}


# Agent 2: Technical Agent
def technical_agent(state: AgentState):
    print("\n[Technical Agent] Working...")

    prompt = f"""
    You are a Kubernetes Technical Architect.

    Based on the research, propose a technical solution.

    Include:
    - Architecture
    - Kubernetes components
    - Deployment approach
    - Security
    - Monitoring

    User question:
    {state['question']}

    Research findings:
    {state['research']}
    """

    response = llm.invoke([
        SystemMessage(content="You are a Kubernetes expert."),
        HumanMessage(content=prompt)
    ])

    return {"technical": response.content}


# Agent 3: Report Agent
def report_agent(state: AgentState):
    print("\n[Report Agent] Working...")

    prompt = f"""
    Prepare a structured technical report.

    Include:
    1. Executive summary
    2. Research findings
    3. Technical architecture
    4. Implementation steps
    5. Conclusion

    User question:
    {state['question']}

    Research:
    {state['research']}

    Technical solution:
    {state['technical']}

    Do not invent unsupported facts.
    """

    response = llm.invoke([
        SystemMessage(content="You are a Technical Report Agent."),
        HumanMessage(content=prompt)
    ])

    return {"report": response.content}


# Build LangGraph
graph = StateGraph(AgentState)

# Register nodes
graph.add_node("coordinator", coordinator)
graph.add_node("research", research_agent)
graph.add_node("technical", technical_agent)
graph.add_node("report", report_agent)

# Define workflow
graph.add_edge(START, "coordinator")
graph.add_edge("coordinator", "research")
graph.add_edge("research", "technical")
graph.add_edge("technical", "report")
graph.add_edge("report", END)

# Compile graph
app = graph.compile()


# =========================================================
# PDF GENERATION
# =========================================================

def save_report_pdf(result, question):

    pdf_file = "multi_agent_report.pdf"

    doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    content = []

    # Title
    content.append(
        Paragraph("Multi-Agent Technical Report", title_style)
    )

    content.append(Spacer(1, 20))

    # User Question
    content.append(
        Paragraph("User Question", heading_style)
    )

    question_text = (
        str(question)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    content.append(
        Paragraph(question_text, normal_style)
    )

    content.append(Spacer(1, 15))

    # Research Agent
    content.append(
        Paragraph("Research Agent", heading_style)
    )

    research_text = (
        str(result["research"])
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )

    content.append(
        Paragraph(research_text, normal_style)
    )

    content.append(Spacer(1, 15))

    # Technical Agent
    content.append(
        Paragraph("Technical Agent", heading_style)
    )

    technical_text = (
        str(result["technical"])
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )

    content.append(
        Paragraph(technical_text, normal_style)
    )

    content.append(Spacer(1, 15))

    # Final Report
    content.append(
        Paragraph("Final Report", heading_style)
    )

    report_text = (
        str(result["report"])
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )

    content.append(
        Paragraph(report_text, normal_style)
    )

    # Build PDF
    doc.build(content)

    print(f"\n[PDF] Report saved as: {pdf_file}")

    return pdf_file


# =========================================================
# SMTP EMAIL
# =========================================================

def send_pdf_email(pdf_file, receiver_email, question):

    sender_email = os.environ["SENDER_EMAIL"]
    sender_password = os.environ["SENDER_APP_PASSWORD"]

    # Create email
    msg = EmailMessage()

    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = "Multi-Agent Technical Report"

    # Email body
    msg.set_content(
        f"""
Hello,

Please find attached the generated Multi-Agent Technical Report.

Question:
{question}

The report was generated automatically using the LangGraph multi-agent system.

Regards,
Multi-Agent System
"""
    )

    # Read PDF
    with open(pdf_file, "rb") as file:

        pdf_data = file.read()

    # Attach PDF
    msg.add_attachment(
        pdf_data,
        maintype="application",
        subtype="pdf",
        filename=pdf_file
    )

    try:

        print("\n[SMTP] Connecting to Gmail...")

        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465
        ) as smtp:

            smtp.login(
                sender_email,
                sender_password
            )

            smtp.send_message(msg)

        print("[SMTP] Email sent successfully!")
        print(f"[SMTP] Receiver: {receiver_email}")

    except smtplib.SMTPAuthenticationError:

        print("\n[SMTP ERROR] Gmail authentication failed.")
        print("Check your SENDER_EMAIL and SENDER_APP_PASSWORD.")

    except smtplib.SMTPException as e:

        print("\n[SMTP ERROR]", e)

    except Exception as e:

        print("\n[ERROR]", e)


# =========================================================
# EXECUTE APPLICATION
# =========================================================

if __name__ == "__main__":

    question = input("Enter your question: ")

    result = app.invoke({
        "question": question,
        "research": "",
        "technical": "",
        "report": ""
    })

    print("\n========== FINAL REPORT ==========")

    print(result["report"])

    # Generate PDF
    pdf_file = save_report_pdf(
        result,
        question
    )

    # Ask receiver email every time
    receiver_email = input(
        "\nEnter receiver email: "
    )

    # Send PDF through SMTP
    send_pdf_email(
        pdf_file,
        receiver_email,
        question
    )

