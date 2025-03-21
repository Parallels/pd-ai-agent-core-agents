from pd_ai_agent_core.messages import Message
from pd_ai_core_agents.background_agents.health_check.vm_health_check_test import (
    VMHealthCheckTest,
    FAILURE_MESSAGE,
    RECOVERY_MESSAGE,
)
from pd_ai_agent_core.parallels_desktop import VirtualMachine
from pd_ai_agent_core.messages import (
    create_error_notification_message,
    create_success_notification_message,
    NotificationAction,
    NotificationActionType,
)
from pd_ai_agent_core.parallels_desktop.execute_on_vm import execute_on_vm
from pd_ai_agent_core.helpers.image import detect_black_screen
import logging
from typing import Tuple
from pd_ai_agent_core.messages import (
    VM_HEALTH_CHECK,
    VM_REBOOT,
    VM_DISABLE_HEALTH_CHECK_TEST,
    VM_SEND_REPORT,
)
from pd_ai_core_agents.common.messages import (
    HEALTH_CHECK_TEST_DETECT_INTERNET_CONNECTION,
)

logger = logging.getLogger(__name__)


class DetectInternetConnectionHealthCheckTest(VMHealthCheckTest):
    def __init__(self, session_id: str, vm: VirtualMachine, count_for_failure: int = 3):
        super().__init__(
            session_id=session_id,
            vm=vm,
            name=HEALTH_CHECK_TEST_DETECT_INTERNET_CONNECTION,
            count_for_failure=count_for_failure,
        )

    async def _check_function(self) -> Tuple[bool, str]:
        execution_result = execute_on_vm(self.vm.id, "ping -c 1 google.com")
        if execution_result.exit_code != 0:
            logger.error(
                f"Error pinging google.com for VM {self.vm.id}: {execution_result.error}"
            )
            return False, "Error pinging the internet"

        return True, ""

    def _failure_message(self) -> Message:
        notification_message = create_error_notification_message(
            session_id=self.session_id,
            channel=self.vm.id,
            message=FAILURE_MESSAGE,
            details=f"We cannot access the internet from the VM {self.vm.name}.\nPlease check the VM if you have internet connectivity.",
            data={
                "vm_id": self.vm.id,
            },
            replace=True,
            actions=[
                NotificationAction(
                    label="Reboot",
                    value=VM_REBOOT,
                    icon="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIwLjA5NjcgMTIuMzU5M0MyMC4wOTY3IDEzLjM2NDUgMTkuOTI3NCAxNC4zMTI0IDE5LjU4ODkgMTUuMjAzQzE5LjI1MDMgMTYuMDkzNyAxOC43NzY0IDE2Ljg5ODMgMTguMTY3IDE3LjYxNzFDMTcuNTYyOCAxOC4zMzA2IDE2Ljg1MTkgMTguOTI0NCAxNi4wMzQyIDE5LjM5ODNDMTUuMjIxNyAxOS44Nzc1IDE0LjMzODkgMjAuMjAzIDEzLjM4NTcgMjAuMzc0OVYyMS4zMjAyQzEzLjM4NTcgMjEuNTE4MSAxMy4zMzg5IDIxLjY2NjYgMTMuMjQ1MSAyMS43NjU1QzEzLjE1NjYgMjEuODY0NSAxMy4wNDIgMjEuOTExNCAxMi45MDE0IDIxLjkwNjJDMTIuNzYwNyAyMS45MDYyIDEyLjYxNDkgMjEuODUxNSAxMi40NjM5IDIxLjc0MjFMOS45MTY5OSAxOS44ODI3QzkuNzQ1MTIgMTkuNzU3NyA5LjY1OTE4IDE5LjYwNjcgOS42NTkxOCAxOS40Mjk2QzkuNjY0MzkgMTkuMjU3NyA5Ljc1MDMzIDE5LjEwOTMgOS45MTY5OSAxOC45ODQzTDEyLjQ3MTcgMTcuMTI0OUMxMi42MTc1IDE3LjAyMDcgMTIuNzYwNyAxNi45NjYgMTIuOTAxNCAxNi45NjA4QzEzLjA0MiAxNi45NTU2IDEzLjE1NjYgMTcuMDA1MSAxMy4yNDUxIDE3LjEwOTNDMTMuMzM4OSAxNy4yMDgyIDEzLjM4NTcgMTcuMzU0MSAxMy4zODU3IDE3LjU0NjhWMTguNTMxMkMxNC4wODg5IDE4LjM3NDkgMTQuNzM5OSAxOC4xMDkzIDE1LjMzODkgMTcuNzM0M0MxNS45Mzc4IDE3LjM1NDEgMTYuNDU4NyAxNi44OTA1IDE2LjkwMTQgMTYuMzQzN0MxNy4zNDQxIDE1Ljc5MTYgMTcuNjg3OCAxNS4xNzk2IDE3LjkzMjYgMTQuNTA3N0MxOC4xODI2IDEzLjgzMDYgMTguMzA3NiAxMy4xMTQ1IDE4LjMwNzYgMTIuMzU5M0MxOC4zMDc2IDExLjUxNTUgMTguMTQ4OCAxMC43MjEzIDE3LjgzMTEgOS45NzY0NkMxNy41MTMzIDkuMjMxNjcgMTcuMDc4NSA4LjU4MDYzIDE2LjUyNjQgOC4wMjMzNEMxNi4zMjMyIDcuNzc4NTUgMTYuMjMyMSA3LjU0NDE3IDE2LjI1MjkgNy4zMjAyMUMxNi4yNzkgNy4wOTYyNiAxNi4zNjQ5IDYuOTExMzYgMTYuNTEwNyA2Ljc2NTUzQzE2LjY3NzQgNi41OTg4NiAxNi44OTEgNi41MTAzMiAxNy4xNTE0IDYuNDk5OUMxNy40MTE4IDYuNDg5NDkgMTcuNjQxIDYuNjAxNDYgMTcuODM4OSA2LjgzNTg0QzE4LjUzNjggNy41Mzg5NiAxOS4wODYzIDguMzY3MDkgMTkuNDg3MyA5LjMyMDIxQzE5Ljg5MzYgMTAuMjY4MSAyMC4wOTY3IDExLjI4MTIgMjAuMDk2NyAxMi4zNTkzWk0zLjc5OTggMTIuMzU5M0MzLjc5OTggMTEuMzU0MSAzLjk2OTA4IDEwLjQwNjIgNC4zMDc2MiA5LjUxNTUzQzQuNjQ2MTYgOC42MTk2OSA1LjEyMDEyIDcuODE1MDEgNS43Mjk0OSA3LjEwMTQ2QzYuMzM4ODcgNi4zODI3MSA3LjA0OTggNS43ODYzNiA3Ljg2MjMgNS4zMTI0QzguNjc0OCA0LjgzMzI0IDkuNTU3NjIgNC41MTAzMiAxMC41MTA3IDQuMzQzNjVWMy4zOTgzNEMxMC41MTA3IDMuMTk1MjEgMTAuNTU1IDMuMDQ0MTcgMTAuNjQzNiAyLjk0NTIxQzEwLjczNzMgMi44NDYyNiAxMC44NTQ1IDIuNzk5MzggMTAuOTk1MSAyLjgwNDU5QzExLjEzNTcgMi44MDQ1OSAxMS4yNzkgMi44NTkyOCAxMS40MjQ4IDIuOTY4NjVMMTMuOTc5NSA0LjgzNTg0QzE0LjE1MTQgNC45NjA4NCAxNC4yMzczIDUuMTExODggMTQuMjM3MyA1LjI4ODk2QzE0LjIzNzMgNS40NjA4NCAxNC4xNTE0IDUuNjA5MjggMTMuOTc5NSA1LjczNDI4TDExLjQyNDggNy41OTM2NUMxMS4yNzkgNy42OTc4MiAxMS4xMzU3IDcuNzUyNTEgMTAuOTk1MSA3Ljc1NzcxQzEwLjg1NDUgNy43NjI5MiAxMC43MzczIDcuNzE2MDUgMTAuNjQzNiA3LjYxNzA5QzEwLjU1NSA3LjUxMjkyIDEwLjUxMDcgNy4zNjQ0OSAxMC41MTA3IDcuMTcxNzhWNi4xODc0QzkuODA3NjIgNi4zNDM2NSA5LjE1NjU4IDYuNjExODggOC41NTc2MiA2Ljk5MjA5QzcuOTU4NjYgNy4zNjcwOSA3LjQzNTIyIDcuODMwNjMgNi45ODczIDguMzgyNzFDNi41NDQ2IDguOTI5NTkgNi4yMDA4NSA5LjU0MTU3IDUuOTU2MDUgMTAuMjE4N0M1LjcxMTI2IDEwLjg5MDUgNS41ODg4NyAxMS42MDQxIDUuNTg4ODcgMTIuMzU5M0M1LjU4ODg3IDEzLjIwMyA1Ljc0NzcyIDEzLjk5NzMgNi4wNjU0MyAxNC43NDIxQzYuMzgzMTQgMTUuNDg2OSA2LjgxODAzIDE2LjEzNTMgNy4zNzAxMiAxNi42ODc0QzcuNTczMjQgMTYuOTM3NCA3LjY2MTc4IDE3LjE3NDQgNy42MzU3NCAxNy4zOTgzQzcuNjA5NyAxNy42MjIzIDcuNTI2MzcgMTcuODA0NiA3LjM4NTc0IDE3Ljk0NTJDNy4yMTkwOCAxOC4xMTE5IDcuMDA1NTMgMTguMjAwNCA2Ljc0NTEyIDE4LjIxMDhDNi40ODQ3IDE4LjIyNjUgNi4yNTI5MyAxOC4xMTcxIDYuMDQ5OCAxNy44ODI3QzUuMzUxODkgMTcuMTc0NCA0LjgwMjQxIDE2LjM0NjMgNC40MDEzNyAxNS4zOTgzQzQuMDAwMzMgMTQuNDQ1MiAzLjc5OTggMTMuNDMyMiAzLjc5OTggMTIuMzU5M1oiIGZpbGw9ImJsYWNrIi8+Cjwvc3ZnPgo=",
                    kind=NotificationActionType.BACKGROUND_MESSAGE,
                    data={
                        "message_type": VM_HEALTH_CHECK,
                        "vm_id": self.vm.id,
                    },
                ),
                NotificationAction(
                    label="Disable Test",
                    value=VM_DISABLE_HEALTH_CHECK_TEST,
                    icon="data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IS0tIFVwbG9hZGVkIHRvOiBTVkcgUmVwbywgd3d3LnN2Z3JlcG8uY29tLCBHZW5lcmF0b3I6IFNWRyBSZXBvIE1peGVyIFRvb2xzIC0tPg0KPHN2ZyB3aWR0aD0iODAwcHgiIGhlaWdodD0iODAwcHgiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4NCjxwYXRoIGQ9Ik0zIDNMMjEgMjFNOS4zNzc0NyAzLjU2MzI1QzEwLjE4NzEgMy4xOTYwNCAxMS4wODI3IDMgMTIgM0MxMy41OTEzIDMgMTUuMTE3NCAzLjU5IDE2LjI0MjYgNC42NDAyQzE3LjM2NzkgNS42OTA0MSAxOCA3LjExNDc5IDE4IDguNkMxOCAxMC4zNTY2IDE4LjI4OTIgMTEuNzc1OSAxOC43MTIgMTIuOTEyMk0xNyAxN0gxNU02LjQ1MzM5IDYuNDY0NTFDNi4xNTY4NiA3LjEzNTQyIDYgNy44NjAxNiA2IDguNkM2IDExLjI4NjIgNS4zMjM4IDEzLjE4MzUgNC41Mjc0NSAxNC40ODY2QzMuNzU2MTYgMTUuNzQ4NiAzLjM3MDUxIDE2LjM3OTcgMy4zODQ4NSAxNi41NDM2QzMuNDAwOTUgMTYuNzI3NyAzLjQzNzI5IDE2Ljc5MjUgMy41ODYwMyAxNi45MDIzQzMuNzE4NDEgMTcgNC4zNDc2MiAxNyA1LjYwNjA1IDE3SDlNOSAxN1YxOEM5IDE5LjY1NjkgMTAuMzQzMSAyMSAxMiAyMUMxMy42NTY5IDIxIDE1IDE5LjY1NjkgMTUgMThWMTdNOSAxN0gxNSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPg0KPC9zdmc+",
                    kind=NotificationActionType.BACKGROUND_MESSAGE,
                    data={
                        "message_type": VM_HEALTH_CHECK,
                        "vm_id": self.vm.id,
                        "test_name": HEALTH_CHECK_TEST_DETECT_INTERNET_CONNECTION,
                    },
                ),
            ],
        )
        return notification_message

    def _recovery_message(self) -> Message:
        notification_message = create_success_notification_message(
            session_id=self.session_id,
            channel=self.vm.id,
            message=RECOVERY_MESSAGE,
            details=f"We can now access the internet from the VM {self.vm.name}. We will keep monitoring it.",
            data={
                "vm_id": self.vm.id,
            },
            replace=True,
        )
        return notification_message
