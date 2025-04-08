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
            icon="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIEdlbmVyYXRvcjogU1ZHIFJlcG8gTWl4ZXIgVG9vbHMgLS0+Cgo8c3ZnCiAgIHdpZHRoPSI4MDBweCIKICAgaGVpZ2h0PSI4MDBweCIKICAgdmlld0JveD0iMCAwIDI0IDI0IgogICBmaWxsPSJub25lIgogICB2ZXJzaW9uPSIxLjEiCiAgIGlkPSJzdmcxIgogICBzb2RpcG9kaTpkb2NuYW1lPSJsaXN0LXN2Z3JlcG8tY29tLnN2ZyIKICAgaW5rc2NhcGU6dmVyc2lvbj0iMS40IChlN2MzZmViMSwgMjAyNC0xMC0wOSkiCiAgIHhtbG5zOmlua3NjYXBlPSJodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy9uYW1lc3BhY2VzL2lua3NjYXBlIgogICB4bWxuczpzb2RpcG9kaT0iaHR0cDovL3NvZGlwb2RpLnNvdXJjZWZvcmdlLm5ldC9EVEQvc29kaXBvZGktMC5kdGQiCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgeG1sbnM6c3ZnPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgPGRlZnMKICAgICBpZD0iZGVmczEiIC8+CiAgPHNvZGlwb2RpOm5hbWVkdmlldwogICAgIGlkPSJuYW1lZHZpZXcxIgogICAgIHBhZ2Vjb2xvcj0iI2ZmZmZmZiIKICAgICBib3JkZXJjb2xvcj0iIzAwMDAwMCIKICAgICBib3JkZXJvcGFjaXR5PSIwLjI1IgogICAgIGlua3NjYXBlOnNob3dwYWdlc2hhZG93PSIyIgogICAgIGlua3NjYXBlOnBhZ2VvcGFjaXR5PSIwLjAiCiAgICAgaW5rc2NhcGU6cGFnZWNoZWNrZXJib2FyZD0iMCIKICAgICBpbmtzY2FwZTpkZXNrY29sb3I9IiNkMWQxZDEiCiAgICAgaW5rc2NhcGU6em9vbT0iMC44OTE4Mzg0MyIKICAgICBpbmtzY2FwZTpjeD0iMzgwLjY3NDMzIgogICAgIGlua3NjYXBlOmN5PSI0ODcuMTk1ODciCiAgICAgaW5rc2NhcGU6d2luZG93LXdpZHRoPSIxMjAwIgogICAgIGlua3NjYXBlOndpbmRvdy1oZWlnaHQ9IjExODYiCiAgICAgaW5rc2NhcGU6d2luZG93LXg9IjAiCiAgICAgaW5rc2NhcGU6d2luZG93LXk9IjI1IgogICAgIGlua3NjYXBlOndpbmRvdy1tYXhpbWl6ZWQ9IjAiCiAgICAgaW5rc2NhcGU6Y3VycmVudC1sYXllcj0ic3ZnMSIgLz4KICA8ZwogICAgIGlkPSJwYXRoMSI+CiAgICA8cGF0aAogICAgICAgc3R5bGU9ImZpbGw6I2ZmYjM4MDtzdHJva2UtbGluZWNhcDpyb3VuZDtzdHJva2UtbGluZWpvaW46cm91bmQiCiAgICAgICBkPSJtIDgsNiAxMyw3LjhlLTQgTSA4LDEyIGwgMTMsOGUtNCBNIDgsMTggbCAxMyw3ZS00IE0gMyw2LjUgaCAxIHYgLTEgSCAzIFogbSAwLDYgaCAxIHYgLTEgSCAzIFogbSAwLDYgaCAxIHYgLTEgSCAzIFoiCiAgICAgICBpZD0icGF0aDIiIC8+CiAgICA8cGF0aAogICAgICAgc3R5bGU9ImZpbGw6IzAwMDAwMDtzdHJva2UtbGluZWNhcDpyb3VuZDtzdHJva2UtbGluZWpvaW46cm91bmQiCiAgICAgICBkPSJNIDMsNC45MDAzOTA2IEEgMC42MDAwNjAwMiwwLjYwMDA2MDAyIDAgMCAwIDIuNDAwMzkwNiw1LjUgdiAxIEEgMC42MDAwNjAwMiwwLjYwMDA2MDAyIDAgMCAwIDMsNy4wOTk2MDk0IEggNCBBIDAuNjAwMDYwMDIsMC42MDAwNjAwMiAwIDAgMCA0LjU5OTYwOTQsNi41IHYgLTEgQSAwLjYwMDA2MDAyLDAuNjAwMDYwMDIgMCAwIDAgNCw0LjkwMDM5MDYgWiBtIDUsMC41IEEgMC42MDAwMDAwMiwwLjYwMDAwMDAyIDAgMCAwIDcuNDAwMzkwNiw2IDAuNjAwMDAwMDIsMC42MDAwMDAwMiAwIDAgMCA4LDYuNTk5NjA5NCBsIDEzLDAuMDAxOTUgQSAwLjYwMDAwMDAyLDAuNjAwMDAwMDIgMCAwIDAgMjEuNTk5NjA5LDYgMC42MDAwMDAwMiwwLjYwMDAwMDAyIDAgMCAwIDIxLDUuNDAwMzkwNiBaIE0gMywxMC45MDAzOTEgQSAwLjYwMDA2MDAyLDAuNjAwMDYwMDIgMCAwIDAgMi40MDAzOTA2LDExLjUgdiAxIEEgMC42MDAwNjAwMiwwLjYwMDA2MDAyIDAgMCAwIDMsMTMuMDk5NjA5IEggNCBBIDAuNjAwMDYwMDIsMC42MDAwNjAwMiAwIDAgMCA0LjU5OTYwOTQsMTIuNSB2IC0xIEEgMC42MDAwNjAwMiwwLjYwMDA2MDAyIDAgMCAwIDQsMTAuOTAwMzkxIFogbSA1LDAuNSBBIDAuNjAwMDAwMDIsMC42MDAwMDAwMiAwIDAgMCA3LjQwMDM5MDYsMTIgMC42MDAwMDAwMiwwLjYwMDAwMDAyIDAgMCAwIDgsMTIuNTk5NjA5IGwgMTMsMC4wMDIgQSAwLjYwMDAwMDAyLDAuNjAwMDAwMDIgMCAwIDAgMjEuNTk5NjA5LDEyIDAuNjAwMDAwMDIsMC42MDAwMDAwMiAwIDAgMCAyMSwxMS40MDAzOTEgWiBtIC01LDUuNSBBIDAuNjAwMDYwMDIsMC42MDAwNjAwMiAwIDAgMCAyLjQwMDM5MDYsMTcuNSB2IDEgQSAwLjYwMDA2MDAyLDAuNjAwMDYwMDIgMCAwIDAgMywxOS4wOTk2MDkgSCA0IEEgMC42MDAwNjAwMiwwLjYwMDA2MDAyIDAgMCAwIDQuNTk5NjA5NCwxOC41IHYgLTEgQSAwLjYwMDA2MDAyLDAuNjAwMDYwMDIgMCAwIDAgNCwxNi45MDAzOTEgWiBtIDUsMC41IEEgMC42MDAwMDAwMiwwLjYwMDAwMDAyIDAgMCAwIDcuNDAwMzkwNiwxOCAwLjYwMDAwMDAyLDAuNjAwMDAwMDIgMCAwIDAgOCwxOC41OTk2MDkgbCAxMywwLjAwMiBBIDAuNjAwMDAwMDIsMC42MDAwMDAwMiAwIDAgMCAyMS41OTk2MDksMTggMC42MDAwMDAwMiwwLjYwMDAwMDAyIDAgMCAwIDIxLDE3LjQwMDM5MSBaIgogICAgICAgaWQ9InBhdGgzIiAvPgogIDwvZz4KPC9zdmc+Cg==",
            functions=[self.get_vms_lists, self.get_vm_details],  # type: ignore
            function_descriptions=[
                AgentFunctionDescriptor(
                    name=self.get_vms_lists.__name__,
                    description="Getting all VMs",
                ),
                AgentFunctionDescriptor(
                    name=self.get_vm_details.__name__,
                    description="Getting VM details",
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
            dictResults = [vm.to_dict() for vm in vmsResult]
            ls.debug(
                session_context["channel"],
                f"Listing VMs, result: {vmsResult}",
            )
            if session_context["session_id"] and ns is not None:
                ns.send_event_sync(
                    session_context["channel"],
                    "vms",
                    "info",
                    dictResults,
                )

            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            return LlmChatAgentResponse(
                status="success",
                message="VMs listed successfully",
                data=dictResults,
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
        except Exception as e:
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
                    vm.to_dict(),
                )

            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            return LlmChatAgentResponse(
                status="success",
                message=f"VM {vm_id} details retrieved successfully",
                data=vm.to_dict(),
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
        except Exception as e:
            ns.send_sync(
                create_clean_agent_function_call_chat_message(
                    session_context["session_id"], session_context["channel"]
                )
            )
            return LlmChatAgentResponse(
                status="error",
                message=f"Failed to get VM {vm_id}: {e}",
                error=str(e),
            )
