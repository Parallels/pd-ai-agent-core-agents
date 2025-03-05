from pd_ai_agent_core.messages import Message
from background_agents.health_check_agent.vm_health_check_test import (
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
from pd_ai_agent_core_agents.messages import (
    HEALTH_CHECK_TEST_DETECT_GUEST_TOOLS,
)

logger = logging.getLogger(__name__)


class DetectGuestToolsHealthCheckTest(VMHealthCheckTest):
    def __init__(self, session_id: str, vm: VirtualMachine, count_for_failure: int = 3):
        super().__init__(
            session_id=session_id,
            vm=vm,
            name="Detect Guest Tools",
            count_for_failure=count_for_failure,
        )

    async def _check_function(self) -> Tuple[bool, str]:
        execution_result = execute_on_vm(self.vm.id, "echo 'hello'")
        if execution_result.exit_code != 0:
            logger.error(
                f"Error getting guest tools for VM {self.vm.id}: {execution_result.error}"
            )
            return False, "Error getting guest tools"

        return True, ""

    def _failure_message(self) -> Message:
        notification_message = create_error_notification_message(
            session_id=self.session_id,
            channel=self.vm.id,
            message=FAILURE_MESSAGE,
            details=f"We cannot access the guest tools of the VM {self.vm.name}.\nPlease check the VM to see if it is still running.",
            data={
                "vm_id": self.vm.id,
            },
            replace=True,
            actions=[
                NotificationAction(
                    label="Reboot",
                    value=VM_REBOOT,
                    icon="restart",
                    kind=NotificationActionType.BACKGROUND_MESSAGE,
                    data={
                        "message_type": VM_HEALTH_CHECK,
                        "vm_id": self.vm.id,
                    },
                ),
                NotificationAction(
                    label="Send Report",
                    value=VM_SEND_REPORT,
                    icon="bug-report",
                    kind=NotificationActionType.BACKGROUND_MESSAGE,
                    data={
                        "message_type": VM_SEND_REPORT,
                        "vm_id": self.vm.id,
                    },
                ),
                NotificationAction(
                    label="Disable Test",
                    value=VM_DISABLE_HEALTH_CHECK_TEST,
                    icon="bell-slash",
                    kind=NotificationActionType.BACKGROUND_MESSAGE,
                    data={
                        "message_type": VM_HEALTH_CHECK,
                        "vm_id": self.vm.id,
                        "test_name": HEALTH_CHECK_TEST_DETECT_GUEST_TOOLS,
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
            details=f"We can access the guest tools of the VM {self.vm.name}. We will keep monitoring it.",
            data={
                "vm_id": self.vm.id,
            },
            replace=True,
        )
        return notification_message
