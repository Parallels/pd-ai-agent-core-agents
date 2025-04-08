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
            icon="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIEdlbmVyYXRvcjogU1ZHIFJlcG8gTWl4ZXIgVG9vbHMgLS0+Cgo8c3ZnCiAgIHdpZHRoPSI4MDBweCIKICAgaGVpZ2h0PSI4MDBweCIKICAgdmlld0JveD0iMCAwIDI0IDI0IgogICBmaWxsPSJub25lIgogICB2ZXJzaW9uPSIxLjEiCiAgIGlkPSJzdmcxIgogICBzb2RpcG9kaTpkb2NuYW1lPSJkaWFncmFtLXN1Y2Nlc3Nvci1zdmdyZXBvLWNvbS5zdmciCiAgIGlua3NjYXBlOnZlcnNpb249IjEuNCAoZTdjM2ZlYjEsIDIwMjQtMTAtMDkpIgogICB4bWxuczppbmtzY2FwZT0iaHR0cDovL3d3dy5pbmtzY2FwZS5vcmcvbmFtZXNwYWNlcy9pbmtzY2FwZSIKICAgeG1sbnM6c29kaXBvZGk9Imh0dHA6Ly9zb2RpcG9kaS5zb3VyY2Vmb3JnZS5uZXQvRFREL3NvZGlwb2RpLTAuZHRkIgogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxkZWZzCiAgICAgaWQ9ImRlZnMxIiAvPgogIDxzb2RpcG9kaTpuYW1lZHZpZXcKICAgICBpZD0ibmFtZWR2aWV3MSIKICAgICBwYWdlY29sb3I9IiNmZmZmZmYiCiAgICAgYm9yZGVyY29sb3I9IiMwMDAwMDAiCiAgICAgYm9yZGVyb3BhY2l0eT0iMC4yNSIKICAgICBpbmtzY2FwZTpzaG93cGFnZXNoYWRvdz0iMiIKICAgICBpbmtzY2FwZTpwYWdlb3BhY2l0eT0iMC4wIgogICAgIGlua3NjYXBlOnBhZ2VjaGVja2VyYm9hcmQ9IjAiCiAgICAgaW5rc2NhcGU6ZGVza2NvbG9yPSIjZDFkMWQxIgogICAgIGlua3NjYXBlOnpvb209IjAuNjMwNjI1IgogICAgIGlua3NjYXBlOmN4PSIzMTcuOTM4NTUiCiAgICAgaW5rc2NhcGU6Y3k9IjY4Ny40MTMyOCIKICAgICBpbmtzY2FwZTp3aW5kb3ctd2lkdGg9IjEyMDAiCiAgICAgaW5rc2NhcGU6d2luZG93LWhlaWdodD0iMTE4NiIKICAgICBpbmtzY2FwZTp3aW5kb3cteD0iMCIKICAgICBpbmtzY2FwZTp3aW5kb3cteT0iMjUiCiAgICAgaW5rc2NhcGU6d2luZG93LW1heGltaXplZD0iMCIKICAgICBpbmtzY2FwZTpjdXJyZW50LWxheWVyPSJzdmcxIiAvPgogIDxwYXRoCiAgICAgc3R5bGU9ImZpbGw6IzAwMDAwMDtzdHJva2UtbGluZWNhcDpyb3VuZDtzdHJva2UtbGluZWpvaW46cm91bmQiCiAgICAgZD0ibSA2LDIuNDAwMzkwNiBjIC0wLjkzMTg4LDAgLTEuNDYwOTYxMywtMC4wMjM1NzggLTEuOTk0MTQwNiwwLjE5NzI2NTcgQyAzLjM2OTAwNjgsMi44NjE0NTUzIDIuODYxNDU1MywzLjM2OTAwNjggMi41OTc2NTYzLDQuMDA1ODU5NCAyLjM3NjgxMjIsNC41MzkwMzg3IDIuNDAwMzkwNiw1LjA2ODEyIDIuNDAwMzkwNiw2IGMgMCwwLjkzMTg4IC0wLjAyMzU3OCwxLjQ2MDk2MTMgMC4xOTcyNjU3LDEuOTk0MTQwNiAwLjI2Mzc5OSwwLjYzNjg1MjYgMC43NzEzNTA1LDEuMTQ0NDA0MSAxLjQwODIwMzEsMS40MDgyMDMyIEMgNC41MzkwMzg3LDkuNjIzMTg3OCA1LjA2ODEyLDkuNTk5NjA5NCA2LDkuNTk5NjA5NCBoIDMgYyAwLjkzMTg4LDAgMS40NjA5NTQsMC4wMjM1NTEgMS45OTQxNDEsLTAuMTk3MjY1NiBDIDExLjYzMDkyOCw5LjEzODU0NSAxMi4xMzg1MjIsOC42MzEwMTcxIDEyLjQwMjM0NCw3Ljk5NDE0MDYgMTIuNjIzMTIsNy40NjA5ODU3IDEyLjU5OTYwOSw2LjkzMTg4IDEyLjU5OTYwOSw2IGMgMCwtMC45MzE4OCAwLjAyMzUxLC0xLjQ2MDk4NTcgLTAuMTk3MjY1LC0xLjk5NDE0MDYgLTAuMDc2MTEsLTAuMTgzNzMxNCAtMC4yOTI0NDgsLTAuMjQ2OTI3NyAtMC40MDYyNSwtMC40MDYyNSBoIDMuODA0Njg3IGMgMS4xMjAxLDAgMS41OTQ5NDIsMC4wMzA1MzcgMS44MzM5ODUsMC4xNTIzNDM3IDAuMjYzNDg0LDAuMTM0MjU1OSAwLjQ3OTA5NCwwLjM0OTg1NzMgMC42MTMyODEsMC42MTMyODEzIDAuMTIxODI4LDAuMjM5MDg0NiAwLjE1MjM0NCwwLjcxMzg4NDMgMC4xNTIzNDQsMS44MzM5ODQzIFYgNy41NTI3MzQ0IEwgMTcuNDIzODI4LDYuNTc2MTcxOSBjIC0wLjIzNDIxNCwtMC4yMzM3MzM3IC0wLjYxMzQ0MiwtMC4yMzM3MzM3IC0wLjg0NzY1NiwwIC0wLjIzMzczNCwwLjIzNDIxNDUgLTAuMjMzNzM0LDAuNjEzNDQxNyAwLDAuODQ3NjU2MiBsIDIsMiBDIDE4LjY4ODU4LDkuNTM2MjU5OCAxOC44NDEwMTUsOS41OTk0ODE4IDE5LDkuNTk5NjA5NCBjIDAuMTU4OTg1LC0xLjI3NmUtNCAwLjMxMTQyLC0wLjA2MzM1IDAuNDIzODI4LC0wLjE3NTc4MTMgbCAyLC0yIGMgMC4yMzM3MzQsLTAuMjM0MjE0NSAwLjIzMzczNCwtMC42MTM0NDE3IDAsLTAuODQ3NjU2MiAtMC4yMzQyMTQsLTAuMjMzNzMzNyAtMC42MTM0NDIsLTAuMjMzNzMzNyAtMC44NDc2NTYsMCBMIDE5LjU5OTYwOSw3LjU1MjczNDQgViA2LjE5OTIxODcgYyAwLC0xLjEyMDA5OTkgMC4wMzA5NywtMS43NjIzNTA4IC0wLjI4MzIwMywtMi4zNzg5MDYyIEMgMTkuMDY3MTkzLDMuMzMxMDc2NCAxOC42Njg4MDMsMi45MzI4MTc5IDE4LjE3OTY4NywyLjY4MzU5MzggMTcuNTYzMTMsMi4zNjk0MjAzIDE2LjkyMDg4MSwyLjQwMDM5MDYgMTUuODAwNzgxLDIuNDAwMzkwNiBIIDkgWiBtIDAsMS4xOTkyMTg4IGggMyBjIDAuOTMxODgsMCAxLjMzMzE0MywwLjAyMzc1OSAxLjUzNTE1NiwwLjEwNzQyMTkgMC4zNDMyMTMsMC4xNDIxODEyIDAuNjE1NjM1LDAuNDE0NTg4OSAwLjc1NzgxMywwLjc1NzgxMjQgQyAxMS4zNzY1OTMsNC42NjY3ODg4IDExLjQwMDM5MSw1LjA2ODEyIDExLjQwMDM5MSw2IGMgMCwwLjkzMTg4IC0wLjAyMzgsMS4zMzMyMTEyIC0wLjEwNzQyMiwxLjUzNTE1NjIgLTAuMTQyMTc4LDAuMzQzMjIzNiAtMC40MTQ2LDAuNjE1NjMxMyAtMC43NTc4MTMsMC43NTc4MTI2IEMgMTAuMzMzMTQzLDguMzc2NjMxNiA5LjkzMTg4LDguNDAwMzkwNiA5LDguNDAwMzkwNiBIIDYgYyAtMC45MzE4OCwwIC0xLjMzMzIzNTYsLTAuMDIzNzg2IC0xLjUzNTE1NjMsLTAuMTA3NDIxOCBDIDQuMTIxNTk2NCw4LjE1MDc4NzggMy44NDkyMTIyLDcuODc4NDAzNiAzLjcwNzAzMTMsNy41MzUxNTYyIDMuNjIzMzk1Myw3LjMzMzIzNTYgMy41OTk2MDk0LDYuOTMxODggMy41OTk2MDk0LDYgYyAwLC0wLjkzMTg4IDAuMDIzNzg2LC0xLjMzMzIzNTYgMC4xMDc0MjE5LC0xLjUzNTE1NjMgQyAzLjg0OTIxMjIsNC4xMjE1OTY0IDQuMTIxNTk2NCwzLjg0OTIxMjIgNC40NjQ4NDM3LDMuNzA3MDMxMyA0LjY2Njc2NDQsMy42MjMzOTUzIDUuMDY4MTIsMy41OTk2MDk0IDYsMy41OTk2MDk0IFogTSA2LDE0LjQwMDM5MSBjIC0wLjkzMTg4LDAgLTEuNDYwOTg1NywtMC4wMjM1MSAtMS45OTQxNDA2LDAuMTk3MjY1IEMgMy4zNjg5ODI5LDE0Ljg2MTQ3OCAyLjg2MTQ1NSwxNS4zNjkwNzIgMi41OTc2NTYzLDE2LjAwNTg1OSAyLjM3Njg0MiwxNi41MzkwMzkgMi40MDAzOTA2LDE3LjA2ODEgMi40MDAzOTA2LDE4IGMgMCwwLjkzMTkgLTAuMDIzNTQ5LDEuNDYwOTYxIDAuMTk3MjY1NywxLjk5NDE0MSAwLjI2Mzc5ODcsMC42MzY3ODcgMC43NzEzMjY2LDEuMTQ0MzgxIDEuNDA4MjAzMSwxLjQwODIwMyBDIDQuNTM5MDE0MywyMS42MjMxMiA1LjA2ODEyLDIxLjU5OTYwOSA2LDIxLjU5OTYwOSBoIDEyIGMgMC45MzE5LDAgMS40NjA5ODUsMC4wMjM0OCAxLjk5NDE0MSwtMC4xOTcyNjUgMC42MzY4MTEsLTAuMjYzODIyIDEuMTQ0MzgxLC0wLjc3MTM5MiAxLjQwODIwMywtMS40MDgyMDMgQyAyMS42MjMwOSwxOS40NjA5ODUgMjEuNTk5NjA5LDE4LjkzMTkgMjEuNTk5NjA5LDE4IGMgMCwtMC45MzE5IDAuMDIzNDgsLTEuNDYwOTg1IC0wLjE5NzI2NSwtMS45OTQxNDEgQyAyMS4xMzg1MjIsMTUuMzY5MDQ4IDIwLjYzMDk1MiwxNC44NjE0NzggMTkuOTk0MTQxLDE0LjU5NzY1NiAxOS40NjA5ODUsMTQuMzc2OTEgMTguOTMxOSwxNC40MDAzOTEgMTgsMTQuNDAwMzkxIFogbSAwLDEuMTk5MjE4IGggMTIgYyAwLjkzMTksMCAxLjMzMzExMiwwLjAyMzc3IDEuNTM1MTU2LDAuMTA3NDIyIDAuMzQzMTg5LDAuMTQyMTc5IDAuNjE1NjM0LDAuNDE0NjI0IDAuNzU3ODEzLDAuNzU3ODEzIDAuMDgzNjUsMC4yMDIwNDQgMC4xMDc0MjIsMC42MDMyNTYgMC4xMDc0MjIsMS41MzUxNTYgMCwwLjkzMTkgLTAuMDIzNzcsMS4zMzMxMTIgLTAuMTA3NDIyLDEuNTM1MTU2IC0wLjE0MjE3OSwwLjM0MzE4OSAtMC40MTQ2MjQsMC42MTU2MzQgLTAuNzU3ODEzLDAuNzU3ODEzIEMgMTkuMzMzMTEyLDIwLjM3NjYyMyAxOC45MzE5LDIwLjQwMDM5MSAxOCwyMC40MDAzOTEgSCA2IGMgLTAuOTMxODgsMCAtMS4zMzMyMTEyLC0wLjAyMzggLTEuNTM1MTU2MywtMC4xMDc0MjIgQyA0LjEyMTYyMDIsMjAuMTUwNzkxIDMuODQ5MjEyNSwxOS44NzgzNjkgMy43MDcwMzEzLDE5LjUzNTE1NiAzLjYyMzM2NTUsMTkuMzMzMTM2IDMuNTk5NjA5NCwxOC45MzE5IDMuNTk5NjA5NCwxOCBjIDAsLTAuOTMxOSAwLjAyMzc1NiwtMS4zMzMxMzYgMC4xMDc0MjE5LC0xLjUzNTE1NiBDIDMuODQ5MjEyNSwxNi4xMjE2MzEgNC4xMjE2MjAyLDE1Ljg0OTIwOSA0LjQ2NDg0MzcsMTUuNzA3MDMxIDQuNjY2Nzg4OCwxNS42MjM0MDcgNS4wNjgxMiwxNS41OTk2MDkgNiwxNS41OTk2MDkgWiBtIDAsMS44MDA3ODIgQyA1LjY2ODkzNDMsMTcuNDAwNjA2IDUuNDAwNjA2MywxNy42Njg5MzQgNS40MDAzOTA2LDE4IDUuNDAwNjA2MywxOC4zMzEwNjYgNS42Njg5MzQzLDE4LjU5OTM5NCA2LDE4LjU5OTYwOSBIIDE4IEMgMTguMzMxMDY2LDE4LjU5OTM5MyAxOC41OTkzOTMsMTguMzMxMDY2IDE4LjU5OTYwOSwxOCAxOC41OTkzOTMsMTcuNjY4OTM0IDE4LjMzMTA2NiwxNy40MDA2MDcgMTgsMTcuNDAwMzkxIFoiCiAgICAgaWQ9InBhdGgxIgogICAgIHNvZGlwb2RpOm5vZGV0eXBlcz0ic2Njc2Njc3NjY3NjY3NjY3NjY2NjY2NjY2NjY3NjY3Nzc3NzY2NzY2Nzc2Njc2Njc3NjY3NjY3NzY2NzY2Nzc3NzY2NzY2Nzc2Njc2Njc2NjY2NjY2MiIC8+Cjwvc3ZnPgo=",
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
