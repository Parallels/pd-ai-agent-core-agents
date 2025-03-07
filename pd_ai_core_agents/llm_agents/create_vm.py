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
from pd_ai_agent_core.helpers import (
    get_context_variable,
)
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    LOGGER_SERVICE_NAME,
    VM_DATASOURCE_SERVICE_NAME,
)
from pd_ai_core_agents.llm_agents.helpers import get_vm_details
from pd_ai_agent_core.parallels_desktop.clone_vm import clone_vm

logger = logging.getLogger(__name__)


def CREATE_VM_PROMPT(context_variables) -> str:
    result = """You are an intelligent and empathetic support agent for Parallels.

You can help with create, or clone a VM based on requirements.

You will receive the requirement from another agent or from the user and you are responsible 
for creating the VM with the correct parameters and return that id to the user.
You will need to know the type of OS, and any dependencies that are needed.
if in the requirements you find any dependencies, summarize them so we can pass them to 
the agent responsible for executing commands on the VM.

Once you have successfully created the VM, do the following:
1. return the id of the VM
2. generate a new input for the triage agent with:
    - what was the previous requirements
    - what need to be done next
    - the id of the VM
3. make sure you generate the correct input for the triage agent.

Once you are done, return to the triage agent.

If the user wants to clone a VM, the user needs to provide the name of the new VM.
If this is not present, you need to ask the user for the name of the new VM.
The new vm name parameter should always be called new_vm_name

3. generate a new input for the triage agent with:
    - what was the previous requirements
    - what need to be done next
    - the id of the VM

"""
    if context_variables is not None:
        result += f"""Use the provided context in JSON format: {json.dumps(context_variables)}\
If the user has provided a vm id, use it to perform the operation on the VM.
If the user has provided a vm name, use it on your responses to the user to identify the VM instead of the vm id.

"""
    return result


CREATE_VM_TRANSFER_INSTRUCTIONS = """
Call this function if the user is asking you to create a VM.
"""


class CreateVmAgent(LlmChatAgent):
    def __init__(self):
        super().__init__(
            name="Create VM Agent",
            instructions=CREATE_VM_PROMPT,
            description="This agent is responsible for creating a VM.",
            functions=[self.create_vm],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.create_vm_tool.__name__,
                    description="Creating a VM...",
                ),
                AgentFunctionDescriptor(
                    name=self.clone_vm_tool.__name__,
                    description="Cloning a VM...",
                ),
            ],
            transfer_instructions=CREATE_VM_TRANSFER_INSTRUCTIONS,
        )

    def create_vm_tool(
        self,
        session_context: dict,
        context_variables: dict,
        new_vm_name: str,
    ) -> LlmChatAgentResponse:
        """Create a new VM based on requirements."""
        # For now, return a mock VM ID as a JSON response
        print("Creating VM", new_vm_name)
        return LlmChatAgentResponse(
            status="success",
            message=f"Created VM with requirements: {new_vm_name}",
        )

    def clone_vm_tool(
        self,
        session_context: dict,
        context_variables: dict,
        vm_id: str,
        new_vm_name: str,
    ) -> LlmChatAgentResponse:
        """Clone a VM based on requirements.
        Args:
            vm_id (str): The ID or name of the virtual machine to clone.
        Returns:
            dict: The result of the cloning the VM.
        """

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
            if not new_vm_name:
                context_new_vm_name = get_context_variable(
                    "new_vm_name", session_context, context_variables
                )
                if not context_new_vm_name:
                    return LlmChatAgentResponse(
                        status="error",
                        message="No new VM name provided",
                    )
                new_vm_name = context_new_vm_name

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
                f"Cloning VM {vm_id} to {new_vm_name}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Cloning VM {vm_id} to {new_vm_name}",
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                    arguments={},
                )
            )
            vm, error = get_vm_details(session_context, context_variables, vm_id)
            if error:
                return error
            if not vm:
                return LlmChatAgentResponse(
                    status="error",
                    message="No vm details provided",
                )
            if vm.state == "running" or vm.state == "suspended" or vm.state == "paused":
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is running, we need it to be stopped",
                )
            clone_result = clone_vm(vm_id, new_vm_name)
            if not clone_result.success:
                return LlmChatAgentResponse(
                    status="error",
                    message=clone_result.message,
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
                message=f"Cloned VM {vm_id}",
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
                f"Failed to clone VM {vm_id}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to clone VM {vm_id}: {e}",
                error=str(e),
            )
