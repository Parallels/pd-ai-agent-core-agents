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
            icon="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4NCjwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIEdlbmVyYXRvcjogU1ZHIFJlcG8gTWl4ZXIgVG9vbHMgLS0+DQo8c3ZnIHdpZHRoPSI4MDBweCIgaGVpZ2h0PSI4MDBweCIgdmlld0JveD0iMCAwIDUxMiA1MTIiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+DQogICAgPHRpdGxlPnN1cHBvcnQ8L3RpdGxlPg0KICAgIDxnIGlkPSJQYWdlLTEiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPg0KICAgICAgICA8ZyBpZD0ic3VwcG9ydCIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDIuNjY2NjY3LCA0Mi42NjY2NjcpIj4NCiAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNzkuNzM0MzU1LDE3NC41MDY2NjcgQzM3My4xMjEwMjIsMTA2LjY2NjY2NyAzMzMuMDE0MzU1LC0yLjEzMTYyODIxZS0xNCAyMDkuMDY3Njg4LC0yLjEzMTYyODIxZS0xNCBDODUuMTIxMDIxNywtMi4xMzE2MjgyMWUtMTQgNDUuMDE0MzU1LDEwNi42NjY2NjcgMzguNDAxMDIxNywxNzQuNTA2NjY3IEMxNS4yMDEyNjMyLDE4My4zMTE1NjkgLTAuMTAxNjQzNDUzLDIwNS41ODU3OTkgMC4wMDA1MDgzMDQyNTksMjMwLjQgTDAuMDAwNTA4MzA0MjU5LDI2MC4yNjY2NjcgQzAuMDAwNTA4MzA0MjU5LDI5My4yNTY0NzUgMjYuNzQ0NTQ2MywzMjAgNTkuNzM0MzU1LDMyMCBDOTIuNzI0MTYzOCwzMjAgMTE5LjQ2NzY4OCwyOTMuMjU2NDc1IDExOS40Njc2ODgsMjYwLjI2NjY2NyBMMTE5LjQ2NzY4OCwyMzAuNCBDMTE5LjM2MDQzMSwyMDYuMTIxNDU2IDEwNC42MTk1NjQsMTg0LjMwNDk3MyA4Mi4xMzQzNTUsMTc1LjE0NjY2NyBDODYuNDAxMDIxNywxMzUuODkzMzMzIDEwNy4zMDc2ODgsNDIuNjY2NjY2NyAyMDkuMDY3Njg4LDQyLjY2NjY2NjcgQzMxMC44Mjc2ODgsNDIuNjY2NjY2NyAzMzEuNTIxMDIyLDEzNS44OTMzMzMgMzM1Ljc4NzY4OCwxNzUuMTQ2NjY3IEMzMTMuMzQ3OTc2LDE4NC4zMjQ4MDYgMjk4LjY4MTU2LDIwNi4xNTU4NTEgMjk4LjY2NzY4OCwyMzAuNCBMMjk4LjY2NzY4OCwyNjAuMjY2NjY3IEMyOTguNzYwMzU2LDI4My4xOTk2NTEgMzExLjkyODYxOCwzMDQuMDcwMTAzIDMzMi41ODc2ODgsMzE0LjAyNjY2NyBDMzIzLjYyNzY4OCwzMzAuODggMzAwLjgwMTAyMiwzNTMuNzA2NjY3IDI0NC42OTQzNTUsMzYwLjUzMzMzMyBDMjMzLjQ3ODg2MywzNDMuNTAyODIgMjExLjc4MDIyNSwzMzYuNzg5MDQ4IDE5Mi45MDY0OTEsMzQ0LjUwOTY1OCBDMTc0LjAzMjc1NywzNTIuMjMwMjY4IDE2My4yNjA0MTgsMzcyLjIyNjgyNiAxNjcuMTk2Mjg2LDM5Mi4yMzUxODkgQzE3MS4xMzIxNTMsNDEyLjI0MzU1MiAxODguNjc1ODg1LDQyNi42NjY2NjcgMjA5LjA2NzY4OCw0MjYuNjY2NjY3IEMyMjUuMTgxNTQ5LDQyNi41Nzc0MjQgMjM5Ljg3MDQ5MSw0MTcuNDE3NDY1IDI0Ny4wNDEwMjIsNDAyLjk4NjY2NyBDMzM4LjU2MTAyMiwzOTIuNTMzMzMzIDM2Ny43ODc2ODgsMzQ1LjM4NjY2NyAzNzYuOTYxMDIyLDMxNy42NTMzMzMgQzQwMS43Nzg0NTUsMzA5LjYxNDMzIDQxOC40Njg4ODUsMjg2LjM1MTUwMiA0MTguMTM0MzU1LDI2MC4yNjY2NjcgTDQxOC4xMzQzNTUsMjMwLjQgQzQxOC4yMzcwMiwyMDUuNTg1Nzk5IDQwMi45MzQxMTQsMTgzLjMxMTU2OSAzNzkuNzM0MzU1LDE3NC41MDY2NjcgWiBNNzYuODAxMDIxNywyNjAuMjY2NjY3IEM3Ni44MDEwMjE3LDI2OS42OTIzMjYgNjkuMTYwMDE0OCwyNzcuMzMzMzMzIDU5LjczNDM1NSwyNzcuMzMzMzMzIEM1MC4zMDg2OTUzLDI3Ny4zMzMzMzMgNDIuNjY3Njg4NCwyNjkuNjkyMzI2IDQyLjY2NzY4ODQsMjYwLjI2NjY2NyBMNDIuNjY3Njg4NCwyMzAuNCBDNDIuNjY3Njg4NCwyMjQuMzAyNjY3IDQ1LjkyMDU3NjUsMjE4LjY2ODQ5OSA1MS4yMDEwMjE2LDIxNS42MTk4MzMgQzU2LjQ4MTQ2NjcsMjEyLjU3MTE2NiA2Mi45ODcyNDM0LDIxMi41NzExNjYgNjguMjY3Njg4NSwyMTUuNjE5ODMzIEM3My41NDgxMzM2LDIxOC42Njg0OTkgNzYuODAxMDIxNywyMjQuMzAyNjY3IDc2LjgwMTAyMTcsMjMwLjQgTDc2LjgwMTAyMTcsMjYwLjI2NjY2NyBaIE0zNDEuMzM0MzU1LDIzMC40IEMzNDEuMzM0MzU1LDIyMC45NzQzNCAzNDguOTc1MzYyLDIxMy4zMzMzMzMgMzU4LjQwMTAyMiwyMTMuMzMzMzMzIEMzNjcuODI2NjgxLDIxMy4zMzMzMzMgMzc1LjQ2NzY4OCwyMjAuOTc0MzQgMzc1LjQ2NzY4OCwyMzAuNCBMMzc1LjQ2NzY4OCwyNjAuMjY2NjY3IEMzNzUuNDY3Njg4LDI2OS42OTIzMjYgMzY3LjgyNjY4MSwyNzcuMzMzMzMzIDM1OC40MDEwMjIsMjc3LjMzMzMzMyBDMzQ4Ljk3NTM2MiwyNzcuMzMzMzMzIDM0MS4zMzQzNTUsMjY5LjY5MjMyNiAzNDEuMzM0MzU1LDI2MC4yNjY2NjcgTDM0MS4zMzQzNTUsMjMwLjQgWiI+DQoNCjwvcGF0aD4NCiAgICAgICAgPC9nPg0KICAgIDwvZz4NCjwvc3ZnPg==",
            functions=[self.tech_support],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.tech_support.__name__,
                    description="Getting technical",
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
