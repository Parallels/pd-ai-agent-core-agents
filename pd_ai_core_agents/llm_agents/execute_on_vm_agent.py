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

logger = logging.getLogger(__name__)


def EXECUTE_ON_VM_PROMPT(context_variables) -> str:
    result = """You are an assistant that executes commands on a VM, but just this, you are unable to do anything else.
You will receive the VM ID and the command to execute.
You will need to execute the command on the VM and return the output.

Once you have executed the command, you need to return a json object with the following keys:
- status: "success" or "error"
- message: the output of the command
- context_variables: the context variables that you have created

If there is nothing else to do you need to pass the rest to the summarize agent so it can summarize
the actions that have been taken.

once you are done, return to the triage agent.


"""
    if context_variables is not None:
        result += f"""Use the provided context in JSON format: {json.dumps(context_variables)}\
If the user has provided a vm id, use it to perform the operation on the VM.
If the user has provided a vm name, use it on your responses to the user to identify the VM instead of the vm id.

"""
    return result


EXECUTE_ON_VM_TRANSFER_INSTRUCTIONS = """
Call this function if the user is asking you to execute a command on a VM.
"""


class ExecuteOnVmAgent(LlmChatAgent):
    def __init__(self):
        super().__init__(
            name="Execute On VM Agent",
            instructions=EXECUTE_ON_VM_PROMPT,
            description="This agent is responsible for executing commands on a VM.",
            functions=[self.execute_on_vm],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.execute_on_vm.__name__,
                    description="Executing command...",
                ),
            ],
            transfer_instructions=EXECUTE_ON_VM_TRANSFER_INSTRUCTIONS,
        )

    def execute_on_vm(
        self, session_context: dict, context_variables: dict, vm_id: str, cmd: str
    ) -> LlmChatAgentResponse:
        """Execute any command on a VM.
        Args:
            vm_id (str): The ID of the VM to execute the command on.
            command (str): The command to execute on the VM.
        Returns:
            dict: The result of the execution.
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

            if not cmd:
                context_cmd = get_context_variable(
                    "command", session_context, context_variables
                )
                if not context_cmd:
                    return LlmChatAgentResponse(
                        status="error",
                        message="No command provided",
                    )
                cmd = context_cmd

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
                f"Executing {cmd} on vm {vm_id} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Executing {cmd} on vm",
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
            vm = data.datasource.get_vm(vm_id)
            if not vm:
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} not found",
                )

            result = execute_on_vm(vm_id, cmd)
            if result.exit_code != 0:
                return LlmChatAgentResponse(
                    status="error",
                    message=f"Failed to execute command {cmd} on vm {vm_id}: {result.error}",
                    error=result.error,
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
                message=result.output,
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
                f"Failed to execute command {cmd} on vm {vm_id}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to execute command {cmd} on vm {vm_id}: {e}",
                error=str(e),
            )
