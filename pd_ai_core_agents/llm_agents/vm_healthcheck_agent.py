from pd_ai_agent_core.core_types.llm_chat_ai_agent import (
    LlmChatAgent,
    LlmChatAgentResponse,
    AgentFunctionDescriptor,
)
from pd_ai_agent_core.services.service_registry import ServiceRegistry
from pd_ai_agent_core.services.notification_service import NotificationService
from pd_ai_agent_core.services.log_service import LogService
from pd_ai_agent_core.messages import create_agent_function_call_chat_message
import logging
from pd_ai_agent_core.parallels_desktop.get_vm_screenshot import get_vm_screenshot
from pd_ai_agent_core.helpers import (
    get_context_variable,
)
from pd_ai_agent_core.helpers.image import detect_black_screen
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    LOGGER_SERVICE_NAME,
)
from pd_ai_core_agents.llm_agents.helpers import get_vm_details
import openai

logger = logging.getLogger(__name__)


def ANALYSE_VM_OCR_LLM_PROMPT(os_version: str, ocr_text: str) -> str:
    return f"""You are an assistant that provides advanced technical support for the user on operating system level.
Your job is to help the user by taking a screenshot of the vm and then using the ocr to get the text of the screenshot.

You will then use the text to help the user get technical support for their operating system.

You should reply with as much detail as possible to help the user get technical support including a lot
of markdown to make it more readable. also if you can add citations to the text, do so.

If you find any useful information and have links to the source, add them to the end of your response.

This is the ocr text:
{ocr_text}

This is the os version:
{os_version}

"""


def VM_HEALTH_CHECK_PROMPT(context_variables) -> str:
    result = f"""You are an assistant that provides advanced technical support for the user on operating system level.
Your job is to help the user by analyzing and taking a screenshot of the vm and then using the ocr to get the text of the screenshot.
You can also use the health check tool to get the health check of the vm.

You will then use the text to help the user get technical support for their operating system and check if everything is working as expected.


"""

    if "vm_id" in context_variables:
        result += f"""
The user has asked to take into acount this context:
{context_variables}
"""
    return result


VM_HEALTH_CHECK_TRANSFER_INSTRUCTIONS = """
Call this function if the user is asking you to analyse the VM, get a health check, or get the ocr of a screenshot or just a screenshot.
    You will need the VM ID or VM Name to do this, if the state is stopped, you will need to start the vm first. check the context or history of the conversation for this information.
"""


class VmHealthCheckAgent(LlmChatAgent):
    def __init__(self):
        super().__init__(
            name="VM Health Check Agent",
            instructions=VM_HEALTH_CHECK_PROMPT,
            description="This agent is responsible for executing commands on a VM.",
            functions=[self.get_health_check_tool],  # type: ignore
            icon="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IS0tIFVwbG9hZGVkIHRvOiBTVkcgUmVwbywgd3d3LnN2Z3JlcG8uY29tLCBHZW5lcmF0b3I6IFNWRyBSZXBvIE1peGVyIFRvb2xzIC0tPg0KPHN2ZyB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4NCjxwYXRoIGQ9Ik0xOC41IDkuMDAwMDJIMTYuNU0xNi41IDkuMDAwMDJMMTQuNSA5LjAwMDAyTTE2LjUgOS4wMDAwMkwxNi41IDdNMTYuNSA5LjAwMDAyTDE2LjUgMTEiIHN0cm9rZT0iIzFDMjc0QyIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPg0KPHBhdGggZD0iTTguOTYxNzMgMTkuMzc4Nkw5LjQzNDMyIDE4Ljc5NjNMOC45NjE3MyAxOS4zNzg2Wk0xMiA1LjU3NDEyTDExLjQ1MjIgNi4wODYzNUMxMS41OTQgNi4yMzgwMyAxMS43OTIzIDYuMzI0MTIgMTIgNi4zMjQxMkMxMi4yMDc3IDYuMzI0MTIgMTIuNDA2IDYuMjM4MDMgMTIuNTQ3OCA2LjA4NjM1TDEyIDUuNTc0MTJaTTE1LjAzODMgMTkuMzc4N0wxNS41MTA5IDE5Ljk2MUwxNS4wMzgzIDE5LjM3ODdaTTEyIDIxTDEyIDIwLjI1TDEyIDIxWk0yLjY1MTU5IDEzLjY4MjFDMi44NjU5NSAxNC4wMzY2IDMuMzI3MDUgMTQuMTUwMSAzLjY4MTQ4IDEzLjkzNThDNC4wMzU5MSAxMy43MjE0IDQuMTQ5NDYgMTMuMjYwMyAzLjkzNTEgMTIuOTA1OUwyLjY1MTU5IDEzLjY4MjFaTTYuNTM3MzMgMTYuMTcwN0M2LjI0ODM2IDE1Ljg3MzkgNS43NzM1MiAxNS44Njc2IDUuNDc2NzYgMTYuMTU2NkM1LjE4IDE2LjQ0NTUgNS4xNzM2OSAxNi45MjA0IDUuNDYyNjcgMTcuMjE3MUw2LjUzNzMzIDE2LjE3MDdaTTIuNzUgOS4zMTc1QzIuNzUgNi40MTI4OSA0LjAxNzY2IDQuNjE3MzEgNS41ODYwMiA0LjAwMzE5QzcuMTUwOTIgMy4zOTA0MyA5LjM0MDM5IDMuODI3NzggMTEuNDUyMiA2LjA4NjM1TDEyLjU0NzggNS4wNjE4OUMxMC4xNTk4IDIuNTA3ODQgNy4zNDkyNCAxLjcwMTg3IDUuMDM5MSAyLjYwNjQ1QzIuNzMyNDIgMy41MDk2NyAxLjI1IDUuOTkyMDkgMS4yNSA5LjMxNzVIMi43NVpNMTUuNTEwOSAxOS45NjFDMTcuMDAzMyAxOC43NDk5IDE4Ljc5MTQgMTcuMTI2OCAyMC4yMTI3IDE1LjMxNEMyMS42MTk2IDEzLjUxOTYgMjIuNzUgMTEuNDM1NCAyMi43NSA5LjMxNzQ3SDIxLjI1QzIxLjI1IDEwLjkyODkgMjAuMzcwNyAxMi42ODE0IDE5LjAzMjMgMTQuMzg4NEMxNy43MDg0IDE2LjA3NyAxNi4wMTU2IDE3LjYxOTcgMTQuNTY1NyAxOC43OTYzTDE1LjUxMDkgMTkuOTYxWk0yMi43NSA5LjMxNzQ3QzIyLjc1IDUuOTkyMDggMjEuMjY3NiAzLjUwOTY2IDE4Ljk2MDkgMi42MDY0NUMxNi42NTA4IDEuNzAxODcgMTMuODQwMiAyLjUwNzg0IDExLjQ1MjIgNS4wNjE4OUwxMi41NDc4IDYuMDg2MzVDMTQuNjU5NiAzLjgyNzc4IDE2Ljg0OTEgMy4zOTA0MiAxOC40MTQgNC4wMDMxOUMxOS45ODIzIDQuNjE3MyAyMS4yNSA2LjQxMjg3IDIxLjI1IDkuMzE3NDdIMjIuNzVaTTguNDg5MTQgMTkuOTYxQzkuNzYwNTggMjAuOTkyOCAxMC42NDIzIDIxLjc1IDEyIDIxLjc1TDEyIDIwLjI1QzExLjI3NzEgMjAuMjUgMTAuODI2OSAxOS45MjYzIDkuNDM0MzIgMTguNzk2M0w4LjQ4OTE0IDE5Ljk2MVpNMTQuNTY1NyAxOC43OTYzQzEzLjE3MzEgMTkuOTI2MyAxMi43MjI5IDIwLjI1IDEyIDIwLjI1TDEyIDIxLjc1QzEzLjM1NzcgMjEuNzUgMTQuMjM5NCAyMC45OTI4IDE1LjUxMDkgMTkuOTYxTDE0LjU2NTcgMTguNzk2M1pNMy45MzUxIDEyLjkwNTlDMy4xODgxMSAxMS42NzA4IDIuNzUgMTAuNDU1IDIuNzUgOS4zMTc1SDEuMjVDMS4yNSAxMC44Mjk3IDEuODI2NDYgMTIuMzE3OSAyLjY1MTU5IDEzLjY4MjFMMy45MzUxIDEyLjkwNTlaTTkuNDM0MzIgMTguNzk2M0M4LjUxNzMxIDE4LjA1MjEgNy40OTg5MyAxNy4xNTgyIDYuNTM3MzMgMTYuMTcwN0w1LjQ2MjY3IDE3LjIxNzFDNi40NzU0OCAxOC4yNTcyIDcuNTM5OTYgMTkuMTkwOCA4LjQ4OTE0IDE5Ljk2MUw5LjQzNDMyIDE4Ljc5NjNaIiBmaWxsPSIjMUMyNzRDIi8+DQo8L3N2Zz4=",
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.get_health_check_tool.__name__,
                    description="Getting the health check of a vm",
                )
            ],
            transfer_instructions=VM_HEALTH_CHECK_TRANSFER_INSTRUCTIONS,
        )

    def analyse_ocr_with_llm(self, os: str, ocr_text: str):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": ANALYSE_VM_OCR_LLM_PROMPT(os, ocr_text),
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error using OpenAI API: {e}")
            return None

    def get_health_check_tool(
        self, session_context: dict, context_variables: dict, vm_id: str
    ) -> LlmChatAgentResponse:
        """Call this function if the user is asking you to get the health check of the vm."""
        try:
            if not vm_id:
                context_vm_id = get_context_variable(
                    "vm_id", session_context, context_variables
                )
                if not context_vm_id:
                    return LlmChatAgentResponse(
                        status="error",
                        message="No VM ID provided",
                    )
                vm_id = context_vm_id
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
                f"Getting ocr from screenshot for vm {vm_id}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_context["session_id"],
                    session_context["channel"],
                    f"Getting screenshot for vm {vm_id}",
                    {},
                )
            )
            vm_details, error = get_vm_details(
                session_context, context_variables, vm_id
            )
            if error:
                return error
            if not vm_details:
                return LlmChatAgentResponse(
                    status="error",
                    message="No vm details provided",
                )
            screenshotResult = get_vm_screenshot(vm_id)
            if not screenshotResult.success:
                return LlmChatAgentResponse(
                    status="error",
                    message="No screenshot provided",
                )
            screenshot = screenshotResult.screenshot
            if screenshot is None:
                return LlmChatAgentResponse(
                    status="error",
                    message="No screenshot provided",
                )

            if detect_black_screen(screenshot):
                logger.error(f"VM {vm_id} has a black screen")
                return LlmChatAgentResponse(
                    status="error",
                    message="VM has a black screen",
                )
            return LlmChatAgentResponse(
                status="success",
                message="VM is healthy",
            )
        except Exception as e:
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to get health check for vm {vm_id}",
                error=str(e),
            )
