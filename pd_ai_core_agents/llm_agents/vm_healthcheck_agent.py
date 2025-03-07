from pd_ai_agent_core.core_types.llm_chat_ai_agent import (
    LlmChatAgent,
    LlmChatResult,
    LlmChatAgentResponse,
    AgentFunctionDescriptor,
)
from pd_ai_agent_core.services.service_registry import ServiceRegistry
from pd_ai_agent_core.services.notification_service import NotificationService
from pd_ai_agent_core.services.log_service import LogService
from pd_ai_agent_core.services.ocr_service import OCRService
from pd_ai_agent_core.services.vm_datasource_service import VmDatasourceService
from pd_ai_agent_core.messages import (
    create_agent_function_call_chat_message,
    create_clean_agent_function_call_chat_message,
)
import logging
from pd_ai_agent_core.parallels_desktop.get_vm_screenshot import get_vm_screenshot
from pd_ai_agent_core.helpers import (
    get_context_variable,
)
from pd_ai_agent_core.helpers.image import detect_black_screen
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    LOGGER_SERVICE_NAME,
    OCR_SERVICE_NAME,
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
            name="Execute On VM Agent",
            instructions=VM_HEALTH_CHECK_PROMPT,
            description="This agent is responsible for executing commands on a VM.",
            functions=[self.execute_on_vm],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.get_health_check_tool.__name__,
                    description="Getting the health check of a vm",
                ),
                AgentFunctionDescriptor(
                    name=self.get_ocr_from_screenshot_tool.__name__,
                    description="Getting the ocr of a screenshot of a vm",
                ),
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

    def get_ocr_from_screenshot_tool(
        self, session_context: dict, context_variables: dict, vm_id: str
    ) -> LlmChatAgentResponse:
        """Call this function if the user is asking you to get the vm screenshot.
        you also this ifg you need to get the ocr of a screenshot"""
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

            os = vm_details.os
            if not os:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="No OS provided",
                )
            screenshot = get_vm_screenshot(vm_id)
            if not screenshot.success:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="No screenshot provided",
                )
            if not screenshot.screenshot:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="No screenshot provided",
                )
            ocr_service = ServiceRegistry.get(
                session_context["session_id"],
                OCR_SERVICE_NAME,
                OCRService,
            )

            ocr_result = ocr_service.ocr(screenshot.screenshot)
            if not ocr_result.text:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="No ocr result provided",
                )
            if ocr_result.average_confidence < 0.5:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="OCR result has an average confidence of less than 50%",
                )
            analysis = self.analyse_ocr_with_llm(os, " ".join(ocr_result.strings))
            if not analysis:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"],
                        session_context["channel"],
                        session_context["linked_message_id"],
                        session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message="No screenshot analysis provided",
                )
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            return LlmChatAgentResponse(status="success", message=analysis)
        except Exception as e:
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            ls.exception(
                session_context["channel"],
                f"Failed to get technical support for vm {vm_id}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to get technical support for vm {vm_id}",
                error=str(e),
            )

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
