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

logger = logging.getLogger(__name__)


def ANALYSE_SUPPORT_PROMPT(os_version: str) -> str:
    return """You are an assistant that provides technical support for the user on operating system level.
Your job is to help the user on getting technical support for their operating system. 
You will need to know the operating system and the version of the operating system.

If you do not know the operating system, you should ask the other agents for help.

You should reply with as much detail as possible to help the user get technical support including a lot
of markdown to make it more readable.

"""


def TECH_SUPPORT_PROMPT(context_variables) -> str:
    result = """You are an assistant that provides technical support for the user on operating system level.
Your job is to help the user on getting technical support for their operating system. 
You will need to know the operating system and the version of the operating system.

You should reply with as much detail as possible to help the user get technical support including a lot
of markdown to make it more readable.
"""
    return result


TECH_SUPPORT_TRANSFER_INSTRUCTIONS = """
Call this function if the user is asking you to provide technical support or any generic questions that are not handled by the other agents.
    for example, how do I run this, or how do I fix this, or check for errors on the vm, or how do I update this, etc.
"""


class TechSupportAgent(LlmChatAgent):
    def __init__(self):
        super().__init__(
            name="Tech Support Agent",
            instructions=TECH_SUPPORT_PROMPT,
            description="This agent is responsible for providing technical support for the user.",
            functions=[self.tech_support],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.tech_support.__name__,
                    description="Getting technical support for a vm",
                ),
            ],
            transfer_instructions=TECH_SUPPORT_TRANSFER_INSTRUCTIONS,
        )

    def analyse_support_with_llm(self, os: str):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": ANALYSE_SUPPORT_PROMPT(os),
                    },
                    {
                        "role": "user",
                        "content": f"OS Version: {os}",
                    },
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error using OpenAI API: {e}")
            return None

    def tech_support(
        self, session_context: dict, context_variables: dict, vm_id: str
    ) -> LlmChatAgentResponse:
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
                f"Getting technical support for vm {vm_id}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Getting technical support for vm",
                    arguments={},
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            data = ServiceRegistry.get(
                session_context["session_id"],
                VM_DATASOURCE_SERVICE_NAME,
                VmDatasourceService,
            )
            vm_details = data.datasource.get_vm(vm_id)
            if not vm_details:
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} not found",
                )
            os = vm_details.os
            if not os:
                return LlmChatAgentResponse(
                    status="error",
                    message="No OS provided",
                )
            analysis = self.analyse_support_with_llm(os)
            if not analysis:
                return LlmChatAgentResponse(
                    status="error",
                    message="No analysis provided",
                )
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                )
            )
            return LlmChatAgentResponse(status="success", message=analysis)
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
                f"Failed to get technical support for vm {vm_id}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to get technical support for vm {vm_id}",
                error=str(e),
            )
