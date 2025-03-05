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
from pd_ai_agent_core.parallels_desktop.get_vm_screenshot import get_vm_screenshot
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
    HEALTH_CHECK_TEST_DETECT_BLACK_SCREEN,
)

logger = logging.getLogger(__name__)


class DetectBlackScreenHealthCheckTest(VMHealthCheckTest):
    def __init__(self, session_id: str, vm: VirtualMachine, count_for_failure: int = 3):
        super().__init__(
            session_id=session_id,
            vm=vm,
            name=HEALTH_CHECK_TEST_DETECT_BLACK_SCREEN,
            count_for_failure=count_for_failure,
        )

    async def _check_function(self) -> Tuple[bool, str]:
        screenshotResult = get_vm_screenshot(self.vm.id)
        if not screenshotResult.success:
            logger.error(
                f"Error getting screenshot for VM {self.vm.id}: {screenshotResult.message}"
            )
            return False, "Error getting screenshot"
        screenshot = screenshotResult.screenshot
        if screenshot is None:
            logger.error(f"Screenshot for VM {self.vm.id} is None")
            return False, "Error getting screenshot"

        if detect_black_screen(screenshot):
            logger.error(f"VM {self.vm.id} has a black screen")
            return False, "VM has a black screen"
        return True, ""

    def _failure_message(self) -> Message:
        notification_message = create_error_notification_message(
            session_id=self.session_id,
            channel=self.vm.id,
            message=FAILURE_MESSAGE,
            details=f"The VM {self.vm.name} is unresponsive, we keep detecting black screens.\nPlease check the VM to see if it is still running.",
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
                        "test_name": HEALTH_CHECK_TEST_DETECT_BLACK_SCREEN,
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
            details=f"The VM {self.vm.name} is responsive again. We will keep monitoring it.",
            data={
                "vm_id": self.vm.id,
            },
            replace=True,
        )
        return notification_message
