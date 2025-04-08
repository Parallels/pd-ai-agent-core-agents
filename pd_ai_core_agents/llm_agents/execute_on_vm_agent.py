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
            icon="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8cGF0aAogICAgZD0iTTExLjU0NzMgMjEuNjI3MUMxMS4yMzA2IDIxLjYyNzEgMTAuOTU0OCAyMS41Mzc3IDEwLjcxOTggMjEuMzU4OUMxMC40ODk5IDIxLjE4MDEgMTAuMzM5MiAyMC45Mzc0IDEwLjI2NzcgMjAuNjMxTDkuOTUzNTcgMTkuMjc0OEw5Ljc0NjcgMTkuMjA1OEw4LjU3NDM3IDE5LjkyNjFDOC4zMDg3NCAyMC4wOTQ2IDguMDI3NzggMjAuMTU4NSA3LjczMTUyIDIwLjExNzdDNy40NDAzNSAyMC4wODE4IDcuMTgyMzggMTkuOTUxNiA2Ljk1NzYyIDE5LjcyNjlMNS44ODQ5MSAxOC42NjE4QzUuNjU1MDQgMTguNDMyIDUuNTIyMjMgMTguMTcxNCA1LjQ4NjQ3IDE3Ljg4MDNDNS40NTA3MSAxNy41ODkxIDUuNTE3MTIgMTcuMzEwNyA1LjY4NTY5IDE3LjA0NTFMNi40MjEyNyAxNS44NzI3TDYuMzUyMzIgMTUuNjgxMkw0Ljk5NjA5IDE1LjM2N0M0LjY5NDcxIDE1LjI5NTUgNC40NTIwNyAxNS4xNDIzIDQuMjY4MTcgMTQuOTA3M0M0LjA4OTM4IDE0LjY3MjMgNCAxNC4zOTkgNCAxNC4wODc0VjEyLjU3OEM0IDEyLjI2NjQgNC4wODkzOCAxMS45OTU2IDQuMjY4MTcgMTEuNzY1OEM0LjQ0Njk1IDExLjUzMDggNC42ODk1OSAxMS4zNzUgNC45OTYwOSAxMS4yOTg0TDYuMzM2OTkgMTAuOTc2Nkw2LjQxMzYgMTAuNzY5N0w1LjY3ODA0IDkuNTk3MzZDNS41MDk0NyA5LjMzNjg0IDUuNDQzMDYgOS4wNjA5OSA1LjQ3ODggOC43Njk4M0M1LjUxNDU2IDguNDc4NjYgNS42NDczNyA4LjIxODE0IDUuODc3MjUgNy45ODgyOEw2Ljk0OTk2IDYuOTE1NTZDNy4xNzQ3MiA2LjY5MDgxIDcuNDMyNjkgNi41NjA1NSA3LjcyMzg2IDYuNTI0NzlDOC4wMTUwMiA2LjQ4MzkyIDguMjkzNDIgNi41NDUyMyA4LjU1OTA0IDYuNzA4NjhMOS43MzkwMyA3LjQzNjZMOS45NTM1NyA3LjM1MjMxTDEwLjI2NzcgNS45OTYwOUMxMC4zMzkyIDUuNjk0NzEgMTAuNDg5OSA1LjQ1NDY0IDEwLjcxOTggNS4yNzU4NUMxMC45NTQ4IDUuMDkxOTUgMTEuMjMwNiA1IDExLjU0NzMgNUgxMy4xMDI4QzEzLjQxOTUgNSAxMy42OTI4IDUuMDkxOTUgMTMuOTIyNiA1LjI3NTg1QzE0LjE1MjUgNS40NTQ2NCAxNC4zMDMyIDUuNjk0NzEgMTQuMzc0NyA1Ljk5NjA5TDE0LjY4ODggNy4zNTIzMUwxNC45MDM0IDcuNDM2NkwxNi4wODMzIDYuNzA4NjhDMTYuMzQ5IDYuNTQ1MjMgMTYuNjI3NCA2LjQ4MzkyIDE2LjkxODUgNi41MjQ3OUMxNy4yMTQ4IDYuNTYwNTUgMTcuNDcyOCA2LjY5MDgxIDE3LjY5MjQgNi45MTU1NkwxOC43NjUyIDcuOTg4MjhDMTguOTg5OSA4LjIxODE0IDE5LjEyMDIgOC40Nzg2NiAxOS4xNTU5IDguNzY5ODNDMTkuMTk2NyA5LjA2MDk5IDE5LjEzMyA5LjMzNjg0IDE4Ljk2NDQgOS41OTczNkwxOC4yMjg4IDEwLjc2OTdMMTguMzEzMSAxMC45NzY2TDE5LjY0NjMgMTEuMjk4NEMxOS45NDc2IDExLjM2OTkgMjAuMTg3OCAxMS41MjMxIDIwLjM2NjYgMTEuNzU4MUMyMC41NTA1IDExLjk5MzEgMjAuNjQyNCAxMi4yNjY0IDIwLjY0MjQgMTIuNTc4VjE0LjA4NzRDMjAuNjQyNCAxNC4zOTkgMjAuNTUwNSAxNC42NzIzIDIwLjM2NjYgMTQuOTA3M0MyMC4xODc4IDE1LjEzNzIgMTkuOTQ3NiAxNS4yOTA0IDE5LjY0NjMgMTUuMzY3TDE4LjI5NzggMTUuNjgxMkwxOC4yMjExIDE1Ljg3MjdMMTguOTU2NyAxNy4wNDUxQzE5LjEyNTMgMTcuMzEwNyAxOS4xODkxIDE3LjU4OTEgMTkuMTQ4MyAxNy44ODAzQzE5LjExMjYgMTguMTcxNCAxOC45ODIyIDE4LjQzMiAxOC43NTc1IDE4LjY2MThMMTcuNjg0NyAxOS43MjY5QzE3LjQ2IDE5Ljk1MTYgMTcuMTk5NSAyMC4wODE4IDE2LjkwMzIgMjAuMTE3N0MxNi42MTIxIDIwLjE1ODUgMTYuMzMzNyAyMC4wOTQ2IDE2LjA2OCAxOS45MjYxTDE0Ljg4ODEgMTkuMjA1OEwxNC42ODg4IDE5LjI3NDhMMTQuMzc0NyAyMC42MzFDMTQuMzAzMiAyMC45Mzc0IDE0LjE1MjUgMjEuMTgwMSAxMy45MjI2IDIxLjM1ODlDMTMuNjkyOCAyMS41Mzc3IDEzLjQxOTUgMjEuNjI3MSAxMy4xMDI4IDIxLjYyNzFIMTEuNTQ3M1pNMTEuNzM4OSAyMC4yNDAySDEyLjkxMTJDMTMuMDMzOCAyMC4yNDAyIDEzLjEwMjggMjAuMTgxNCAxMy4xMTgxIDIwLjA2NEwxMy41Nzc4IDE4LjE3OTFDMTMuODMzMiAxOC4xMjI4IDE0LjA3MDggMTguMDQ4OSAxNC4yOTA0IDE3Ljk1NjlDMTQuNTEwMSAxNy44NTk5IDE0LjcxNDQgMTcuNzUyNSAxNC45MDM0IDE3LjYzNUwxNi41NTA3IDE4LjY0NjVDMTYuNjQ3OCAxOC43MTI5IDE2Ljc0MjMgMTguNzAyNyAxNi44MzQyIDE4LjYxNTlMMTcuNjQ2NSAxNy43OTZDMTcuNzI4MiAxNy43MjQ0IDE3LjczNTkgMTcuNjMyNSAxNy42Njk0IDE3LjUyMDFMMTYuNjU4IDE1Ljg4MDRDMTYuNzY1NCAxNS42OTY1IDE2Ljg2NSAxNS40OTIyIDE2Ljk1NjkgMTUuMjY3NEMxNy4wNTQgMTUuMDQyNyAxNy4xMzA1IDE0LjgxMDMgMTcuMTg2NyAxNC41NzAyTDE5LjA3OTMgMTQuMTE4MUMxOS4xOTY3IDE0LjA5NzcgMTkuMjU1NSAxNC4wMjYxIDE5LjI1NTUgMTMuOTAzNVYxMi43NTQyQzE5LjI1NTUgMTIuNjM2NyAxOS4xOTY3IDEyLjU2NTIgMTkuMDc5MyAxMi41Mzk3TDE3LjE5NDQgMTIuMDg3NkMxNy4xMzMgMTEuODMyMiAxNy4wNTE0IDExLjU4OTUgMTYuOTQ5MSAxMS4zNTk3QzE2Ljg1MjEgMTEuMTI5OCAxNi43NTc2IDEwLjkzMzEgMTYuNjY1NiAxMC43Njk3TDE3LjY3NzEgOS4xMjIzQzE3Ljc0ODYgOS4wMTUwMiAxNy43NDEgOC45MTc5NyAxNy42NTQxIDguODMxMTRMMTYuODQxOSA4LjAzNDI2QzE2Ljc1NSA3Ljk1MjUzIDE2LjY1OCA3LjkzOTc2IDE2LjU1MDcgNy45OTU5NUwxNC45MDM0IDguOTk5NzFDMTQuNzE0NCA4Ljg5MjQzIDE0LjUwNzUgOC43OTI4MiAxNC4yODI3IDguNzAwODdDMTQuMDYzMSA4LjYwMzgyIDEzLjgyODEgOC41MjQ2NCAxMy41Nzc4IDguNDYzMzVMMTMuMTE4MSA2LjU2MzExQzEzLjEwMjggNi40NDU2MSAxMy4wMzM4IDYuMzg2ODcgMTIuOTExMiA2LjM4Njg3SDExLjczODlDMTEuNjExMiA2LjM4Njg3IDExLjUzNzEgNi40NDU2MSAxMS41MTY3IDYuNTYzMTFMMTEuMDcyMyA4LjQ0ODAyQzEwLjgyNzEgOC41MDkzMSAxMC41ODQ0IDguNTkxMDQgMTAuMzQ0MyA4LjY5MzIxQzEwLjEwOTQgOC43OTAyNiA5LjkwNTA1IDguODg5ODcgOS43MzEzNiA4Ljk5MjA0TDguMDgzOTggNy45OTU5NUM3Ljk4MTgxIDcuOTM5NzYgNy44ODczMSA3Ljk0OTk3IDcuODAwNDcgOC4wMjY1OUw2Ljk4MDYyIDguODMxMTRDNi44OTg4OSA4LjkxNzk3IDYuODkxMjIgOS4wMTUwMiA2Ljk1NzYyIDkuMTIyM0w3Ljk2OTA0IDEwLjc2OTdDNy44ODIyMSAxMC45MzMxIDcuNzg3NzEgMTEuMTI5OCA3LjY4NTU0IDExLjM1OTdDNy41ODMzOCAxMS41ODk1IDcuNTA0MTkgMTEuODMyMiA3LjQ0OCAxMi4wODc2TDUuNTYzMDkgMTIuNTM5N0M1LjQ0NTYxIDEyLjU2NTIgNS4zODY4NyAxMi42MzY3IDUuMzg2ODcgMTIuNzU0MlYxMy45MDM1QzUuMzg2ODcgMTQuMDI2MSA1LjQ0NTYxIDE0LjA5NzcgNS41NjMwOSAxNC4xMTgxTDcuNDQ4IDE0LjU2MjVDNy41MDkzMSAxNC44MDc3IDcuNTg4NDggMTUuMDQyNyA3LjY4NTU0IDE1LjI2NzRDNy43ODI1OSAxNS40ODcxIDcuODgyMjEgMTUuNjkxNCA3Ljk4NDM4IDE1Ljg4MDRMNi45NjUyOSAxNy41Mjc4QzYuOTAzOTkgMTcuNjM1IDYuOTExNjUgMTcuNzI3IDYuOTg4MjcgMTcuODAzNkw3LjgwODE0IDE4LjYxNTlDNy45MDAwOSAxOC43MDI3IDcuOTkyMDMgMTguNzE1NSA4LjA4Mzk4IDE4LjY1NDFMOS43MzkwMyAxNy42MzVDOS45MjgwMyAxNy43NTI1IDEwLjEzNDkgMTcuODU5OSAxMC4zNTk3IDE3Ljk1NjlDMTAuNTg5NSAxOC4wNDg5IDEwLjgyNDUgMTguMTIyOCAxMS4wNjQ2IDE4LjE3OTFMMTEuNTE2NyAyMC4wNjRDMTEuNTM3MSAyMC4xODE0IDExLjYxMTIgMjAuMjQwMiAxMS43Mzg5IDIwLjI0MDJaTTEyLjMyMTIgMTYuMjI1MkMxMS43OSAxNi4yMjUyIDExLjMwMjEgMTYuMDk1IDEwLjg1NzcgMTUuODM0NEMxMC40MTg0IDE1LjU2ODggMTAuMDY2IDE1LjIxNjMgOS44MDAzMiAxNC43NzdDOS41Mzk4MSAxNC4zMzc3IDkuNDA5NTUgMTMuODQ5OSA5LjQwOTU1IDEzLjMxMzZDOS40MDk1NSAxMi43ODIzIDkuNTM5ODEgMTIuMjk3IDkuODAwMzIgMTEuODU3N0MxMC4wNjYgMTEuNDE4NCAxMC40MTg0IDExLjA2ODUgMTAuODU3NyAxMC44MDhDMTEuMzAyMSAxMC41NDc1IDExLjc5IDEwLjQxNzIgMTIuMzIxMiAxMC40MTcyQzEyLjg1NzYgMTAuNDE3MiAxMy4zNDU0IDEwLjU0NzUgMTMuNzg0NyAxMC44MDhDMTQuMjI0IDExLjA2ODUgMTQuNTczOSAxMS40MTg0IDE0LjgzNDQgMTEuODU3N0MxNS4wOTQ5IDEyLjI5NyAxNS4yMjUyIDEyLjc4MjMgMTUuMjI1MiAxMy4zMTM2QzE1LjIyNTIgMTMuODQ0OCAxNS4wOTQ5IDE0LjMzMjYgMTQuODM0NCAxNC43NzdDMTQuNTczOSAxNS4yMjE0IDE0LjIyNCAxNS41NzM5IDEzLjc4NDcgMTUuODM0NEMxMy4zNDU0IDE2LjA5NSAxMi44NTc2IDE2LjIyNTIgMTIuMzIxMiAxNi4yMjUyWk0xMi4zMjEyIDE0LjkwNzNDMTIuNjA3MyAxNC45MDczIDEyLjg3MDMgMTQuODM1OCAxMy4xMTA0IDE0LjY5MjhDMTMuMzUwNSAxNC41NDQ2IDEzLjUzOTUgMTQuMzUwNSAxMy42Nzc0IDE0LjExMDRDMTMuODIwNSAxMy44NzAzIDEzLjg5MiAxMy42MDQ3IDEzLjg5MiAxMy4zMTM2QzEzLjg5MiAxMy4wMjI0IDEzLjgyMDUgMTIuNzU5MyAxMy42Nzc0IDEyLjUyNDNDMTMuNTM5NSAxMi4yODQzIDEzLjM1MDUgMTIuMDkyNyAxMy4xMTA0IDExLjk0OTdDMTIuODcwMyAxMS44MDY2IDEyLjYwNzMgMTEuNzM1MSAxMi4zMjEyIDExLjczNTFDMTIuMDMgMTEuNzM1MSAxMS43NjQ0IDExLjgwNjYgMTEuNTI0MyAxMS45NDk3QzExLjI4NDMgMTIuMDkyNyAxMS4wOTI3IDEyLjI4NDMgMTAuOTQ5NyAxMi41MjQzQzEwLjgwNjYgMTIuNzU5MyAxMC43MzUxIDEzLjAyMjQgMTAuNzM1MSAxMy4zMTM2QzEwLjczNTEgMTMuNjA5OCAxMC44MDY2IDEzLjg3OCAxMC45NDk3IDE0LjExODFDMTEuMDkyNyAxNC4zNTgyIDExLjI4NDMgMTQuNTQ5NyAxMS41MjQzIDE0LjY5MjhDMTEuNzY0NCAxNC44MzU4IDEyLjAzIDE0LjkwNzMgMTIuMzIxMiAxNC45MDczWiIKICAgIGZpbGw9ImJsYWNrIiAvPgo8L3N2Zz4=",
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
