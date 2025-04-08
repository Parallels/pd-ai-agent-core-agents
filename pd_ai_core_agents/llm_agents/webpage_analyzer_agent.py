from pd_ai_agent_core.core_types.llm_chat_ai_agent import (
    LlmChatAgent,
    LlmChatResult,
    LlmChatAgentResponse,
    AgentFunctionDescriptor,
)
from pd_ai_agent_core.services.service_registry import ServiceRegistry
from pd_ai_agent_core.services.notification_service import NotificationService
from pd_ai_agent_core.services.log_service import LogService

from pd_ai_agent_core.services.vm_datasource_service import VmDatasourceService
from pd_ai_agent_core.messages import (
    create_agent_function_call_chat_message,
    create_clean_agent_function_call_chat_message,
)
import json
import logging
from pd_ai_agent_core.parallels_desktop.get_vms import get_vm
from pd_ai_agent_core.parallels_desktop.execute_on_vm import execute_on_vm
from pd_ai_agent_core.helpers import (
    get_context_variable,
)
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    LOGGER_SERVICE_NAME,
    VM_DATASOURCE_SERVICE_NAME,
)
import openai
import requests

logger = logging.getLogger(__name__)


def ANALYSE_WEB_PAGE_LLM_PROMPT(context_variables) -> str:
    result = """You are a expert in analyzing webpages and code in webpages for what the user is trying to achieve.
You will need to analyse the webpage and provide a summary of the content and the user's intent.
For example if the user is trying to create a new project, you will need to analyse the webpage and provide a summary of the content and the user's intent.

You will need to return a simple object that follows a coherent structure that can be used by other agents like the triage agent:

for example:
imagine the user intent is to create a new project, you will need to analyse the webpage and provide a summary of:
- os
- languages
- dependencies
- llm_summary
- a code structure with files and files content from what you have analyzed

When generating the code structure, make sure to include all the files and files content from what you have analyzed.
In some cases you might need to join multiple code blocks together to form a complete file.
Try to create a project that will work on the users request.

Make sure to return a good summary of the webpage content and the user's intent. so be extra descriptive.
"""

    if context_variables is not None:
        result += f"""Use the provided context in JSON format: {json.dumps(context_variables)}\
If the user has provided a vm id, use it to perform the operation on the VM.
If the user has provided a vm name, use it on your responses to the user to identify the VM instead of the vm id.

"""

    return result


def ANALYSE_WEB_PAGE_PROMPT(context_variables) -> str:
    result = """You are a expert in analyzing webpages and code in webpages.
You will need to analyze the webpage and code in the webpage and provide a summary of the content and the user's intent.
"""
    if context_variables is not None:
        result += f"""Use the provided context in JSON format: {json.dumps(context_variables)}\
If the user has provided a vm id, use it to perform the operation on the VM.
If the user has provided a vm name, use it on your responses to the user to identify the VM instead of the vm id.

"""
    return result


ANALYSE_WEB_PAGE_TRANSFER_INSTRUCTIONS = """
Call this function if the user is asking you to analyse a webpage or code in a webpage.
"""


class WebpageAnalyzerAgent(LlmChatAgent):
    def __init__(self):
        super().__init__(
            name="Webpage Analyzer Agent",
            instructions=ANALYSE_WEB_PAGE_PROMPT,
            description="This agent is responsible for analyzing webpages for what the user is trying to achieve.",
            icon="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIEdlbmVyYXRvcjogU1ZHIFJlcG8gTWl4ZXIgVG9vbHMgLS0+Cgo8c3ZnCiAgIHdpZHRoPSI4MDBweCIKICAgaGVpZ2h0PSI4MDBweCIKICAgdmlld0JveD0iMCAwIDYwIDYwIgogICB2ZXJzaW9uPSIxLjEiCiAgIGlkPSJzdmc3IgogICBzb2RpcG9kaTpkb2NuYW1lPSJjb2RlLWNvZGluZy1kZXZlbG9wbWVudC1wcm9ncmFtbWluZy13ZWItd2VicGFnZS1zdmdyZXBvLWNvbS5zdmciCiAgIGlua3NjYXBlOnZlcnNpb249IjEuNCAoZTdjM2ZlYjEsIDIwMjQtMTAtMDkpIgogICB4bWxuczppbmtzY2FwZT0iaHR0cDovL3d3dy5pbmtzY2FwZS5vcmcvbmFtZXNwYWNlcy9pbmtzY2FwZSIKICAgeG1sbnM6c29kaXBvZGk9Imh0dHA6Ly9zb2RpcG9kaS5zb3VyY2Vmb3JnZS5uZXQvRFREL3NvZGlwb2RpLTAuZHRkIgogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxzb2RpcG9kaTpuYW1lZHZpZXcKICAgICBpZD0ibmFtZWR2aWV3NyIKICAgICBwYWdlY29sb3I9IiNmZmZmZmYiCiAgICAgYm9yZGVyY29sb3I9IiMwMDAwMDAiCiAgICAgYm9yZGVyb3BhY2l0eT0iMC4yNSIKICAgICBpbmtzY2FwZTpzaG93cGFnZXNoYWRvdz0iMiIKICAgICBpbmtzY2FwZTpwYWdlb3BhY2l0eT0iMC4wIgogICAgIGlua3NjYXBlOnBhZ2VjaGVja2VyYm9hcmQ9IjAiCiAgICAgaW5rc2NhcGU6ZGVza2NvbG9yPSIjZDFkMWQxIgogICAgIGlua3NjYXBlOnpvb209IjAuNjMwNjI1IgogICAgIGlua3NjYXBlOmN4PSIzMjIuNjk1NzQiCiAgICAgaW5rc2NhcGU6Y3k9IjM4Ny43MTA2IgogICAgIGlua3NjYXBlOndpbmRvdy13aWR0aD0iMTIwMCIKICAgICBpbmtzY2FwZTp3aW5kb3ctaGVpZ2h0PSIxMTg2IgogICAgIGlua3NjYXBlOndpbmRvdy14PSIwIgogICAgIGlua3NjYXBlOndpbmRvdy15PSIyNSIKICAgICBpbmtzY2FwZTp3aW5kb3ctbWF4aW1pemVkPSIwIgogICAgIGlua3NjYXBlOmN1cnJlbnQtbGF5ZXI9IkxheWVyXzIiIC8+CiAgPGRlZnMKICAgICBpZD0iZGVmczEiPgogICAgPHN0eWxlCiAgICAgICBpZD0ic3R5bGUxIj4uY2xzLTF7ZmlsbDpub25lO30uY2xzLTJ7ZmlsbDojM2QzZDYzO308L3N0eWxlPgogIDwvZGVmcz4KICA8dGl0bGUKICAgICBpZD0idGl0bGUxIiAvPgogIDxnCiAgICAgZGF0YS1uYW1lPSJMYXllciAyIgogICAgIGlkPSJMYXllcl8yIj4KICAgIDxyZWN0CiAgICAgICBjbGFzcz0iY2xzLTEiCiAgICAgICBoZWlnaHQ9IjYwIgogICAgICAgd2lkdGg9IjYwIgogICAgICAgaWQ9InJlY3QxIgogICAgICAgeD0iMCIKICAgICAgIHk9IjAiIC8+CiAgICA8Y2lyY2xlCiAgICAgICBjbGFzcz0iY2xzLTIiCiAgICAgICBjeD0iOS4zMjk5OTk5IgogICAgICAgY3k9IjE1Ljc0IgogICAgICAgcj0iMSIKICAgICAgIGlkPSJjaXJjbGUxIiAvPgogICAgPGNpcmNsZQogICAgICAgY2xhc3M9ImNscy0yIgogICAgICAgY3g9IjE2IgogICAgICAgY3k9IjE1Ljc0IgogICAgICAgcj0iMSIKICAgICAgIGlkPSJjaXJjbGUyIiAvPgogICAgPGNpcmNsZQogICAgICAgY2xhc3M9ImNscy0yIgogICAgICAgY3g9IjEyLjY3IgogICAgICAgY3k9IjE1Ljc0IgogICAgICAgcj0iMSIKICAgICAgIGlkPSJjaXJjbGUzIiAvPgogICAgPHBhdGgKICAgICAgIGNsYXNzPSJjbHMtMiIKICAgICAgIGQ9Im0gMjAsMzAuMjUgMywtMyBhIDEuMDA0MDkxNiwxLjAwNDA5MTYgMCAxIDAgLTEuNDIsLTEuNDIgbCAtMy43MiwzLjczIGEgMSwxIDAgMCAwIDAsMS40MSBsIDMuNzIsMy43MyBhIDEsMSAwIDAgMCAxLjQyLDAgMSwxIDAgMCAwIDAsLTEuNDEgeiIKICAgICAgIGlkPSJwYXRoMyIgLz4KICAgIDxwYXRoCiAgICAgICBjbGFzcz0iY2xzLTIiCiAgICAgICBkPSJtIDM0LjMyLDM0LjY4IGEgMSwxIDAgMCAwIDEuNDEsMCBMIDM5LjQ2LDMxIGEgMSwxIDAgMCAwIDAsLTEuNDEgbCAtMy43MywtMy43MyBhIDEuMDAwNTYyMywxLjAwMDU2MjMgMCAwIDAgLTEuNDEsMS40MiBsIDMsMyAtMywzIGEgMSwxIDAgMCAwIDAsMS40IHoiCiAgICAgICBpZD0icGF0aDQiIC8+CiAgICA8cGF0aAogICAgICAgY2xhc3M9ImNscy0yIgogICAgICAgZD0ibSAzMSwyNi4xNSBhIDEsMSAwIDAgMCAtMS4zNywwLjM3IEwgMjUuOTQsMzMgYSAxLDEgMCAxIDAgMS43MywxIEwgMzEuNCwyNy41NSBBIDEsMSAwIDAgMCAzMSwyNi4xNSBaIgogICAgICAgaWQ9InBhdGg1IiAvPgogICAgPHBhdGgKICAgICAgIGNsYXNzPSJjbHMtMiIKICAgICAgIGQ9Im0gNTUuODYsNDAgYSAwLjI5LDAuMjkgMCAwIDEgLTAuMjcsLTAuMTkgMTAuNDMsMTAuNDMgMCAwIDAgLTEuNjcsLTIuODkgMC4yOSwwLjI5IDAgMCAxIDAsLTAuMzMgMi4zLDIuMyAwIDAgMCAtMC44NCwtMy4xNCBMIDUyLjMzLDMzIFYgMTQuNzQgYSAzLDMgMCAwIDAgLTMsLTMgSCA4IGEgMywzIDAgMCAwIC0zLDMgdiAyNS4zNCBhIDIuMzQsMi4zNCAwIDAgMCAyLjMzLDIuMzMgSCAyMSBsIC0xLDMuNyBoIC0xLjMzIGEgNC4zMyw0LjMzIDAgMCAwIC00LjIxLDMuMzMgSCAxMiBhIDEsMSAwIDAgMCAwLDIgaCAyNS40NyBhIDIuMTIsMi4xMiAwIDAgMCAwLjA3LDAuNzEgMi4yNSwyLjI1IDAgMCAwIDEuMDcsMS40IGwgMi4xMSwxLjIyIGEgMi4zLDIuMyAwIDAgMCAzLjE0LC0wLjg1IDAuMzEsMC4zMSAwIDAgMSAwLjMxLC0wLjE0IDkuNzQsOS43NCAwIDAgMCAzLjMyLDAgMC4zLDAuMyAwIDAgMSAwLjMxLDAuMTUgMi4zLDIuMyAwIDAgMCAyLDEuMTQgMi4yNywyLjI3IDAgMCAwIDEuMTUsLTAuMyBsIDIuMTEsLTEuMjIgYSAyLjI5LDIuMjkgMCAwIDAgMS4wNywtMS40IDIuMjYsMi4yNiAwIDAgMCAtMC4yMywtMS43NCAwLjMsMC4zIDAgMCAxIDAsLTAuMzQgMTAuNTIsMTAuNTIgMCAwIDAgMS42NywtMi44OCAwLjI4LDAuMjggMCAwIDEgMC4yNywtMC4xOSAyLjMxLDIuMzEgMCAwIDAgMi4zLC0yLjMgViA0Mi4yNiBBIDIuMywyLjMgMCAwIDAgNTUuODYsNDAgWiBNIDcsMTQuNzQgYSAxLDEgMCAwIDEgMSwtMSBoIDQxLjMzIGEgMSwxIDAgMCAxIDEsMSB2IDMgSCA3IFogTSA3LjMzLDQwLjQxIEEgMC4zMywwLjMzIDAgMCAxIDcsNDAuMDggViAxOS43NCBIIDUwLjMzIFYgMzIgYSAyLjI5LDIuMjkgMCAwIDAgLTIuNTMsMSB2IDAgYSAwLjI5LDAuMjkgMCAwIDEgLTAuMywwLjEzIDEwLjI3LDEwLjI3IDAgMCAwIC0zLjMyLDAgMC4yOCwwLjI4IDAgMCAxIC0wLjMsLTAuMTMgdiAwIGEgMi4yOSwyLjI5IDAgMCAwIC0zLjE0LC0wLjg0IGwgLTIuMTEsMS4yMiBhIDIuMjQsMi4yNCAwIDAgMCAtMS4wNywxLjM5IDIuMzIsMi4zMiAwIDAgMCAwLjIyLDEuNzUgMC4yNywwLjI3IDAgMCAxIDAsMC4zMyAxMC40MywxMC40MyAwIDAgMCAtMS42NywyLjg5IDAuMjksMC4yOSAwIDAgMSAtMC4yNywwLjE5IDIuMjksMi4yOSAwIDAgMCAtMS4zNiwwLjQ1IHogTSAzNCw0Ni4xMSBIIDIyIGwgMSwtMy43IGggMTAuNSB2IDIuMjkgYSAyLjMyLDIuMzIgMCAwIDAgMC41LDEuNDEgeiBtIC0xNS4zMiwyIGggMTcuOCBhIDExLDExIDAgMCAwIDAuNzgsMS4zMyBIIDE2LjU3IGEgMi4zMywyLjMzIDAgMCAxIDIuMSwtMS4zMyB6IE0gNTYuMTYsNDQuNyBhIDAuMywwLjMgMCAwIDEgLTAuMywwLjMgMi4zMSwyLjMxIDAgMCAwIC0yLjE1LDEuNDggOC40Niw4LjQ2IDAgMCAxIC0xLjM0LDIuMzMgMi4zMSwyLjMxIDAgMCAwIC0wLjIxLDIuNiAwLjI2LDAuMjYgMCAwIDEgMCwwLjIyIDAuMjksMC4yOSAwIDAgMSAtMC4xNCwwLjE5IEwgNDkuOTQsNTMgYSAwLjMsMC4zIDAgMCAxIC0wLjQxLC0wLjEyIDIuMzEsMi4zMSAwIDAgMCAtMi4zNywtMS4xMSA3LjkyLDcuOTIgMCAwIDEgLTIuNjYsMCAyLjMxLDIuMzEgMCAwIDAgLTIuMzcsMS4xMiAwLjMxLDAuMzEgMCAwIDEgLTAuNDEsMC4xMSBsIC0yLjExLC0xLjIyIGEgMC4yOSwwLjI5IDAgMCAxIC0wLjE0LC0wLjE5IDAuMjYsMC4yNiAwIDAgMSAwLC0wLjIyIDIuMzMsMi4zMyAwIDAgMCAtMC4yMSwtMi42IEEgOC4zNSw4LjM1IDAgMCAxIDM3LjkxLDQ2LjQ0IDIuMjgsMi4yOCAwIDAgMCAzNS44LDQ1IDAuMywwLjMgMCAwIDEgMzUuNSw0NC43IHYgLTIuNDQgYSAwLjI5LDAuMjkgMCAwIDEgMC4zLC0wLjMgMi4zLDIuMyAwIDAgMCAyLjE0LC0xLjQ4IDguMjEsOC4yMSAwIDAgMSAxLjM1LC0yLjMzIDIuMzMsMi4zMyAwIDAgMCAwLjIxLC0yLjYgMC4yOCwwLjI4IDAgMCAxIDAsLTAuMjMgMC4yNiwwLjI2IDAgMCAxIDAuMTQsLTAuMTggbCAyLjExLC0xLjIyIGEgMC4yOCwwLjI4IDAgMCAxIDAuMTUsMCAwLjMsMC4zIDAgMCAxIDAuMjUsMC4xNCB2IDAgYSAyLjMyLDIuMzIgMCAwIDAgMi4zNywxLjEyIDcuOTIsNy45MiAwIDAgMSAyLjY2LDAgMi4zMywyLjMzIDAgMCAwIDIuMzUsLTEuMTggdiAwIGEgMC4yOSwwLjI5IDAgMCAxIDAuNCwtMC4xIGwgMi4xMSwxLjIyIGEgMC4yNiwwLjI2IDAgMCAxIDAuMTQsMC4xOCAwLjI4LDAuMjggMCAwIDEgMCwwLjIzIDIuMywyLjMgMCAwIDAgMC4yMSwyLjU5IDguNTEsOC41MSAwIDAgMSAxLjM0LDIuMzQgMi4zMiwyLjMyIDAgMCAwIDIuMTMsMS41NCAwLjI5LDAuMjkgMCAwIDEgMC4zLDAuMyB6IgogICAgICAgaWQ9InBhdGg2IiAvPgogICAgPHBhdGgKICAgICAgIGNsYXNzPSJjbHMtMiIKICAgICAgIGQ9Im0gNDUuODMsMzguOTQgYSA0LjU0LDQuNTQgMCAxIDAgNC41NCw0LjU0IDQuNTQsNC41NCAwIDAgMCAtNC41NCwtNC41NCB6IG0gMCw3LjA4IEEgMi41NCwyLjU0IDAgMSAxIDQ4LjM3LDQzLjQ4IDIuNTQsMi41NCAwIDAgMSA0NS44Myw0NiBaIgogICAgICAgaWQ9InBhdGg3IiAvPgogIDwvZz4KPC9zdmc+Cg==",
            functions=[self.analyze_webpage_tool],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.analyze_webpage_tool.__name__,
                    description="Analyzing webpage...",
                ),
            ],
            transfer_instructions=ANALYSE_WEB_PAGE_TRANSFER_INSTRUCTIONS,
        )

    def fetch_webpage(self, url: str):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching the webpage: {e}")
            return None

    def analyse_page_with_llm(self, context_variables: dict, html_content: str):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": ANALYSE_WEB_PAGE_LLM_PROMPT(context_variables),
                    },
                    {
                        "role": "user",
                        "content": f"This is the webpage content: {html_content}",
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error using OpenAI API: {e}")
            return None

    def analyze_webpage_tool(
        self, session_context: dict, context_variables: dict, url: str
    ) -> LlmChatAgentResponse:
        """Analyse a webpage.
        Args:
            url (str): The URL of the webpage to analyse.
        Returns:
            dict: The result of the execution.
        """
        try:
            if not url:
                context_url = get_context_variable(
                    "url", session_context, context_variables
                )
                if not context_url:
                    return LlmChatAgentResponse(
                        status="error",
                        message="No URL provided",
                    )
                url = context_url

            ns = ServiceRegistry.get(
                session_context["session_id"],
                NOTIFICATION_SERVICE_NAME,
                NotificationService,
            )
            ls = ServiceRegistry.get(
                session_context["session_id"], LOGGER_SERVICE_NAME, LogService
            )
            ls.info(
                session_context["channel"],
                f"Analyzing webpage {url} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Analyzing webpage {url}",
                    arguments={},
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            html_content = self.fetch_webpage(url)
            if not html_content:
                return LlmChatAgentResponse(
                    status="error",
                    message=f"Webpage {url} not found",
                )

            result = self.analyse_page_with_llm(context_variables, html_content)
            if not result:
                return LlmChatAgentResponse(
                    status="error",
                    message=f"Failed to analyse webpage {url}",
                )
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            return LlmChatAgentResponse(
                status="success",
                message=result,
            )
        except Exception as e:
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            ls.exception(
                session_context["channel"],
                f"Failed to analyse webpage {url}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to analyse webpage {url}",
                error=str(e),
            )
