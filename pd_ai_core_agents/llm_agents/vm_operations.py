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

from pd_ai_agent_core.helpers import (
    get_context_variable,
)
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    LOGGER_SERVICE_NAME,
    VM_DATASOURCE_SERVICE_NAME,
)
from pd_ai_core_agents.llm_agents.helpers import get_vm_details
from pd_ai_agent_core.parallels_desktop.set_vm_state import set_vm_state
from pd_ai_agent_core.parallels_desktop.delete_vm import delete_vm
from pd_ai_agent_core.parallels_desktop.models.set_vm_state_result import (
    VirtualMachineState,
)

logger = logging.getLogger(__name__)


def VM_OPERATION_PROMPT(context_variables) -> str:
    result = """You are an intelligent and empathetic support agent for Parallels.
You will always need to be provided with the id of the VM and the operation to perform. 
This should be a property called "vm_id.

You can help with start, stop, suspend, resume, pause, restart and delete a VM.

You will receive the id of the VM and you need to return the result of the operation.

You need to be polite and generate markdown code blocks when needed.


"""
    if context_variables is not None:
        result += f"""Use the provided context in JSON format: {json.dumps(context_variables)}\
If the user has provided a vm id, use it to perform the operation on the VM.
If the user has provided a vm name, use it on your responses to the user to identify the VM instead of the vm id.

"""
    return result


VM_OPERATION_TRANSFER_INSTRUCTIONS = """
Call this function if the user is asking you to start, stop, suspend, resume, pause, or delete a VM.
    You will need the VM ID or VM Name to do this. check the context or history of the conversation for this information.
"""


class VmOperationsAgent(LlmChatAgent):
    def __init__(self):
        super().__init__(
            name="Vm Operations Agent",
            instructions=VM_OPERATION_PROMPT,
            description="This agent is responsible for starting, stopping, suspending, resuming, pausing, or deleting a VM.",
            functions=[self.start_vm_tool, self.stop_vm_tool, self.suspend_vm_tool, self.resume_vm_tool, self.pause_vm_tool, self.delete_vm_tool],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.start_vm_tool.__name__,
                    description="Starting a VM",
                ),
                AgentFunctionDescriptor(
                    name=self.stop_vm_tool.__name__,
                    description="Stopping a VM",
                ),
                AgentFunctionDescriptor(
                    name=self.suspend_vm_tool.__name__,
                    description="Suspending a VM",
                ),
                AgentFunctionDescriptor(
                    name=self.resume_vm_tool.__name__,
                    description="Resuming a VM",
                ),
                AgentFunctionDescriptor(
                    name=self.pause_vm_tool.__name__,
                    description="Pausing a VM",
                ),
                AgentFunctionDescriptor(
                    name=self.delete_vm_tool.__name__,
                    description="Deleting a VM",
                ),
                AgentFunctionDescriptor(
                    name=self.restart_vm_tool.__name__,
                    description="Restarting a VM",
                ),
                AgentFunctionDescriptor(
                    name=self.get_os_info_tool.__name__,
                    description="Getting OS info for a VM",
                ),
            ],
            transfer_instructions=VM_OPERATION_TRANSFER_INSTRUCTIONS,
        )

    def start_vm_tool(
        self, session_context: dict, context_variables: dict, vm_id
    ) -> LlmChatAgentResponse:
        """Start a specific VM.
        Args:
            vm_id (str): The ID or name of the virtual machine to start.
        Returns:
            dict: The result of the starting the VM.
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
                f"Starting VM {vm_id} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Starting VM",
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                    arguments={},
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

            if vm_details.state == "running":
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is already running",
                )
            if vm_details.state == "suspended" or vm_details.state == "paused":
                operation_result = set_vm_state(
                    vm_id=vm_id, state=VirtualMachineState.RESUME
                )
                if not operation_result:
                    return LlmChatAgentResponse(
                        status="error",
                        message=f"Failed to resume VM {vm_id}",
                    )
                else:
                    return LlmChatAgentResponse(
                        status="success",
                        message=f"VM {vm_id} was suspended or paused and has been resumed",
                    )
            if vm_details.state == "stopped":
                operation_result = set_vm_state(
                    vm_id=vm_id, state=VirtualMachineState.START
                )
                if not operation_result:
                    return LlmChatAgentResponse(
                        status="error",
                        message=f"Failed to start VM {vm_id}",
                    )
                else:
                    return LlmChatAgentResponse(
                        status="success",
                        message=f"VM {vm_id} was stopped and has been started",
                    )
            else:
                return LlmChatAgentResponse(
                    status="error", message=f"VM {vm_id} could not be started"
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
                f"Failed to parse VM {vm_id} output: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to start VM {vm_id}: {e}",
            )

    def stop_vm_tool(
        self, session_context: dict, context_variables: dict, vm_id
    ) -> LlmChatAgentResponse:
        """Stop a specific VM.
        Args:
            vm_id (str): The ID or name of the virtual machine to stop.
        Returns:
            dict: The result of the starting the VM.
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
                f"Stopping VM {vm_id} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Stopping VM",
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                    arguments={},
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

            if vm_details.state == "stopped":
                return LlmChatAgentResponse(
                    status="success",
                    message=f"VM {vm_id} is already stopped",
                )
            if vm_details.state == "suspended" or vm_details.state == "paused":
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is suspended or paused and it cannot be stopped, you need to resume it first",
                )
            if vm_details.state == "running":
                operation_result = set_vm_state(
                    vm_id=vm_id, state=VirtualMachineState.STOP
                )
                if not operation_result:
                    return LlmChatAgentResponse(
                        status="error",
                        message=f"Failed to stop VM {vm_id}",
                    )
                else:
                    return LlmChatAgentResponse(
                        status="success",
                        message=f"VM {vm_id} was running and has been stopped",
                    )
            else:
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} could not be stopped",
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
                f"Failed to stop VM {vm_id}: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to stop VM {vm_id}: {e}",
            )

    def suspend_vm_tool(
        self, session_context: dict, context_variables: dict, vm_id
    ) -> LlmChatAgentResponse:
        """Suspend a specific VM.
        Args:
            vm_id (str): The ID or name of the virtual machine to suspend.
        Returns:
            dict: The result of the suspending the VM.
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
                f"Suspending VM {vm_id} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Suspending VM",
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                    arguments={},
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
            if vm_details.state == "suspended":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is already suspended",
                )
            if vm_details.state == "running":
                operation_result = set_vm_state(
                    vm_id=vm_id, state=VirtualMachineState.SUSPEND
                )
                if not operation_result:
                    ns.send_sync(
                        create_clean_agent_function_call_chat_message(
                            session_id=session_context["session_id"],
                            channel=session_context["channel"],
                            linked_message_id=session_context["linked_message_id"],
                            is_partial=session_context["is_partial"],
                        )
                    )
                    return LlmChatAgentResponse(
                        status="error",
                        message=f"Failed to suspend VM {vm_id}",
                    )
                else:
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
                        message=f"VM {vm_id} was running and has been suspended",
                    )
            else:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} could not be suspended as it is {vm_details.state}",
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
                f"Failed to suspend VM {vm_id}: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to suspend VM {vm_id}: {e}",
            )

    def resume_vm_tool(
        self, session_context: dict, context_variables: dict, vm_id
    ) -> LlmChatAgentResponse:
        """Resume a specific VM.
        Args:
            vm_id (str): The ID or name of the virtual machine to resume.
        Returns:
            dict: The result of the resuming the VM.
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
                f"Resuming VM {vm_id} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Resuming VM",
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                    arguments={},
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

            if vm_details.state == "running":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is already running",
                )
            if vm_details.state == "suspended":
                operation_result = set_vm_state(
                    vm_id=vm_id, state=VirtualMachineState.RESUME
                )
                if not operation_result:
                    ns.send_sync(
                        create_clean_agent_function_call_chat_message(
                            session_id=session_context["session_id"],
                            channel=session_context["channel"],
                            linked_message_id=session_context["linked_message_id"],
                            is_partial=session_context["is_partial"],
                        )
                    )
                    return LlmChatAgentResponse(
                        status="error",
                        message=f"Failed to resume VM {vm_id}",
                    )
                else:
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
                        message=f"VM {vm_id} was suspended and has been resumed",
                    )
            else:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} could not be resumed as it is {vm_details.state}",
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
                f"Failed to resume VM {vm_id}: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to resume VM {vm_id}: {e}",
            )

    def pause_vm_tool(
        self, session_context: dict, context_variables: dict, vm_id
    ) -> LlmChatAgentResponse:
        """Pause a specific VM.
        Args:
            vm_id (str): The ID or name of the virtual machine to pause.
        Returns:
            dict: The result of the pausing the VM.
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
                f"Pausing VM {vm_id} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Pausing VM",
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                    arguments={},
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
            if vm_details.state == "running":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is already running",
                )
            if vm_details.state == "paused":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is already paused",
                )
            if vm_details.state == "running":
                operation_result = set_vm_state(
                    vm_id=vm_id, state=VirtualMachineState.PAUSE
                )
                if not operation_result:
                    ns.send_sync(
                        create_clean_agent_function_call_chat_message(
                            session_id=session_context["session_id"],
                            channel=session_context["channel"],
                            linked_message_id=session_context["linked_message_id"],
                            is_partial=session_context["is_partial"],
                        )
                    )
                    return LlmChatAgentResponse(
                        status="error",
                        message=f"Failed to pause VM {vm_id}",
                    )
                else:
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
                        message=f"VM {vm_id} was running and has been paused",
                    )
            else:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} could not be paused as it is {vm_details.state}",
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
                f"Failed to pause VM {vm_id}: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to pause VM {vm_id}: {e}",
            )

    def delete_vm_tool(
        self, session_context: dict, context_variables: dict, vm_id
    ) -> LlmChatAgentResponse:
        """Delete a specific VM.
        Args:
            vm_id (str): The ID or name of the virtual machine to delete.
        Returns:
            dict: The result of the deleting the VM.
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
                f"Deleting VM {vm_id} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Deleting VM",
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                    arguments={},
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

            if vm_details.state == "running":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is running, you need to stop it first before deleting it",
                )
            if vm_details.state == "suspended":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is suspended, you need to resume it first before deleting it",
                )
            if vm_details.state == "paused":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is paused, you need to resume it first before deleting it",
                )
            if vm_details.state == "stopped":
                operation_result = delete_vm(vm_id=vm_id)
                if not operation_result:
                    ns.send_sync(
                        create_clean_agent_function_call_chat_message(
                            session_id=session_context["session_id"],
                            channel=session_context["channel"],
                            linked_message_id=session_context["linked_message_id"],
                            is_partial=session_context["is_partial"],
                        )
                    )
                    return LlmChatAgentResponse(
                        status="error",
                        message=f"Failed to delete VM {vm_id}",
                    )
                else:
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
                        message=f"VM {vm_id} was stopped and has been deleted",
                    )
            else:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} could not be deleted as it is {vm_details.state}",
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
                f"Failed to delete VM {vm_id}: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to delete VM {vm_id}: {e}",
            )

    def restart_vm_tool(
        self, session_context: dict, context_variables: dict, vm_id
    ) -> LlmChatAgentResponse:
        """Restart a specific VM.
        Args:
            vm_id (str): The ID or name of the virtual machine to restart.
        Returns:
            dict: The result of the restarting the VM.
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
                f"Restarting VM {vm_id} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Restarting VM",
                    linked_message_id=session_context["linked_message_id"],
                    is_partial=session_context["is_partial"],
                    arguments={},
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
            if vm_details.state == "running":
                stop_result = set_vm_state(
                    vm_id=vm_id,
                    state=VirtualMachineState.STOP,
                )
                if stop_result:
                    start_result = set_vm_state(
                        vm_id=vm_id,
                        state=VirtualMachineState.START,
                    )
                    if start_result:
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
                            message=f"VM {vm_id} was running and has been restarted",
                        )
                    else:
                        ns.send_sync(
                            create_clean_agent_function_call_chat_message(
                                session_id=session_context["session_id"],
                                channel=session_context["channel"],
                                linked_message_id=session_context["linked_message_id"],
                                is_partial=session_context["is_partial"],
                            )
                        )
                        return LlmChatAgentResponse(
                            status="error",
                            message=f"Failed to start VM {vm_id}",
                        )
                else:
                    ns.send_sync(
                        create_clean_agent_function_call_chat_message(
                            session_id=session_context["session_id"],
                            channel=session_context["channel"],
                            linked_message_id=session_context["linked_message_id"],
                            is_partial=session_context["is_partial"],
                        )
                    )
                    return LlmChatAgentResponse(
                        status="error",
                        message=f"Failed to stop VM {vm_id}",
                    )
            if vm_details.state == "suspended":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is suspended, you need to resume it first before restarting it",
                )
            if vm_details.state == "paused":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is paused, you need to resume it first before restarting it",
                )
            if vm_details.state == "stopped":
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_id=session_context["session_id"],
                        channel=session_context["channel"],
                        linked_message_id=session_context["linked_message_id"],
                        is_partial=session_context["is_partial"],
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} is stopped, you need to start it first before restarting it",
                )
            else:
                ns.send_sync(
                    create_clean_agent_function_call_chat_message(
                        session_context["session_id"], session_context["channel"]
                    )
                )
                return LlmChatAgentResponse(
                    status="error",
                    message=f"VM {vm_id} could not be restarted as it is {vm_details.state}",
                )
        except Exception as e:

            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            ls.exception(
                session_context["channel"],
                f"Failed to restart VM {vm_id}: {e}",
                e,
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to restart VM {vm_id}: {e}",
            )

    def get_os_info_tool(
        self, session_context: dict, context_variables: dict, vm_id, os: str
    ) -> LlmChatAgentResponse:
        """Get the OS info of a specific VM."""
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
                f"Getting OS info for VM {vm_id} with args {session_context}, {context_variables}",
            )
            ns.send_sync(
                create_agent_function_call_chat_message(
                    session_id=session_context["session_id"],
                    channel=session_context["channel"],
                    name=f"Getting OS info for VM",
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
                message=f"OS info for VM {vm_id}: {vm_details.os}",
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
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to get OS info for VM {vm_id}: {e}",
            )
