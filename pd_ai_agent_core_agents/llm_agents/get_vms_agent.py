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
import subprocess
import logging
from pd_ai_agent_core.parallels_desktop.get_vms import get_vms, get_vm
from pd_ai_agent_core.helpers import (
    get_context_variable,
)
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    LOGGER_SERVICE_NAME,
    VM_DATASOURCE_SERVICE_NAME,
)

logger = logging.getLogger(__name__)


def GET_VMS_AGENT_PROMPT(context_variables) -> str:
    result = """You are an intelligent and empathetic support agent for Parallels.

You can help with getting details about VMs. For example:
- Get the OS version of a VM
- Get the the info about the VM, like name, id, status, ip, mac, memory, cpu, home, os, etc.
- List all VMs with their details


You will need to summarize and respond to the user in a natural language way.
Try to be as accurate as possible and be concise and always be polite and generate markdown code blocks when needed.
What is important to return will be the following:
- name
- id
- status
- ip
- mac
- memory
- cpu
- home
- os

Always use this markdown format template to respond to the user:

If the user asks for a list of VMs, you need to return the following template:
### VM Demo 1
- **Name**: VM Demo 1
- **ID**: 37df58ee-8c21-4465-843e-e8d5f935911b
- **Status**: Suspended
- **IP**: Not available
- **MAC**: 001C4284370A
- **Memory**: 12288 MB
- **CPU**: 4
- **Home**: /Volumes/local_storage_m2/Parallels/AI Development Package v1-t1.pvm/
- **OS**: Ubuntu
### VM Demo 2
- **Name**: VM Demo 2
- **ID**: 37df58ee-8c21-4465-843e-e8d5f935911d
- **Status**: Suspended
- **IP**: Not available
- **MAC**: 001C4284370A
- **Memory**: 12288 MB
- **CPU**: 4
- **Home**: Not available
- **OS**: Ubuntu

If the user asks for a specific VM, or for some particular information about a VM, you need to return a human readable message with the information requested.
"""
    if context_variables is not None:
        result += f"""Use the provided context in JSON format: {json.dumps(context_variables)}\
If the user has provided a vm id, use it to perform the operation on the VM.
If the user has provided a vm name, use it on your responses to the user to identify the VM instead of the vm id.

"""
    return result


GET_VMS_AGENT_TRANSFER_INSTRUCTIONS = """
Call this function if the user is asking you to list all VMs, or list a specific VM or list the details of a specific VM or all VMs.

You can also call this function if the user is asking you to get the OS version of a VM.
"""


class GetVmsAgent(LlmChatAgent):
    def __init__(self):
        super().__init__(
            name="VM List Agent",
            instructions=GET_VMS_AGENT_PROMPT,
            description="This agent is responsible for listing all VMs, or listing a specific VM or listing the details of a specific VM or all VMs.",
            functions=[self.get_vms_lists, self.get_vm_details],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.get_vms_lists.__name__,
                    description="Getting all VMs",
                ),
                AgentFunctionDescriptor(
                    name=self.get_vm_details.__name__,
                    description="Getting details of a specific VM",
                ),
            ],
            transfer_instructions=GET_VMS_AGENT_TRANSFER_INSTRUCTIONS,
        )

    def get_vms_lists(
        self, session_context: dict, context_variables: dict
    ) -> LlmChatAgentResponse:
        """List or get details of all VMs.
        Args:
            session_context (dict): The context of the session.
            context_variables (dict): The context of the context.
            vm_id (str): The ID or name of the virtual machine to get details of.
        Returns:
            AgentResponse: The result of the getting details.
        """
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
            f"Listing all VMs with args {session_context}, {context_variables}",
        )
        ns.send_sync(
            create_agent_function_call_chat_message(
                session_id=session_context["session_id"],
                channel=session_context["channel"],
                name="Listing all VMs",
                linked_message_id=session_context["linked_message_id"],
                is_partial=session_context["is_partial"],
                arguments={},
            )
        )
        data = ServiceRegistry.get(
            session_context["session_id"],
            VM_DATASOURCE_SERVICE_NAME,
            VmDatasourceService,
        )
        try:
            vmsResult = data.datasource.get_all_vms()
            vmsResultDict = json.loads(json.dumps(vmsResult))
            ls.debug(
                session_context["channel"],
                f"Listing VMs, result: {vmsResult}",
            )
            if session_context["session_id"] and ns is not None:
                ns.send_event_sync(
                    session_context["channel"],
                    "vms",
                    "info",
                    vmsResultDict,
                )

            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            return LlmChatAgentResponse(
                status="success",
                message="VMs listed successfully",
                data=vmsResultDict,
            )
        except subprocess.CalledProcessError as e:
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to list VMs: {e}",
                error=str(e),
            )
        except json.JSONDecodeError as e:
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to parse VM list output: {e}",
                error=str(e),
            )

    def get_vm_details(
        self, session_context: dict, context_variables: dict, vm_id: str
    ) -> LlmChatAgentResponse:
        """Get details of a specific VM.
        Args:
            vm_id (str): The ID or name of the virtual machine to get details of.
        Returns:
            AgentResponse: The result of the getting details.
        """

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
            f"Listing VM with args {session_context}, {context_variables}, {vm_id}",
        )
        ns.send_sync(
            create_agent_function_call_chat_message(
                session_id=session_context["session_id"],
                channel=session_context["channel"],
                name=f"Getting details for VM {vm_id}",
                linked_message_id=session_context["linked_message_id"],
                is_partial=session_context["is_partial"],
                arguments={},
            )
        )

        if not vm_id:
            context_vm_id = get_context_variable(
                "vm_id", session_context, context_variables
            )
            if not context_vm_id:
                return LlmChatAgentResponse(
                    status="error",
                    message="No VM ID provided",
                    error="No VM ID provided",
                )
            vm_id = context_vm_id
        try:
            data = ServiceRegistry.get(
                session_context["session_id"],
                VM_DATASOURCE_SERVICE_NAME,
                VmDatasourceService,
            )
            vm = data.datasource.get_vm(vm_id)
            vmDict = json.loads(json.dumps(vm))
            if not vm:
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} not found",
                    error="VM not found",
                )
            ls.debug(
                session_context["channel"],
                f"Listing VM, result: {vm}",
            )
            if session_context["session_id"] and ns is not None:
                ns.send_event_sync(
                    session_context["channel"],
                    "vm",
                    "info",
                    vmDict,
                )

            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            return LlmChatAgentResponse(
                status="success",
                message=f"VM {vm_id} details retrieved successfully",
                data=vmDict,
            )
        except subprocess.CalledProcessError as e:
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            ls.exception(
                session_context["channel"],
                f"Failed to list VM {vm_id}: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to list VM {vm_id}: {e}",
                error=str(e),
            )
        except json.JSONDecodeError as e:
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            ls.exception(
                session_context["channel"],
                f"Failed to parse VM {vm_id} output: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to parse VM {vm_id} output: {e}",
                error=str(e),
            )
