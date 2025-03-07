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
            functions=[self.analyze_webpage_tool],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.analyze_webpage_tool.__name__,
                    description="Analysing webpage...",
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
