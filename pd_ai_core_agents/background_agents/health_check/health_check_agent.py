import logging
from pd_ai_agent_core.helpers.image import detect_black_screen
from pd_ai_agent_core.parallels_desktop.get_vm_screenshot import get_vm_screenshot
from pd_ai_agent_core.messages import (
    VM_HEALTH_CHECK,
    VM_DISABLE_HEALTH_CHECK_TEST,
)
from pd_ai_agent_core.core_types.background_agent import BackgroundAgent
from pd_ai_agent_core.messages import BackgroundMessage
from pd_ai_agent_core.parallels_desktop.datasource import VirtualMachineDataSource
from pd_ai_agent_core.services.notification_service import NotificationService
from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
    LOGGER_SERVICE_NAME,
)
from pd_ai_agent_core.services.service_registry import ServiceRegistry

from datetime import datetime, timedelta
from pd_ai_agent_core.services.log_service import LogService


from pd_ai_core_agents.background_agents.health_check.datasource.health_check_datasource import (
    HealthCheckDataSource,
)
from pd_ai_core_agents.background_agents.health_check.vm_health_check import (
    VmHealthCheck,
)
from pd_ai_agent_core.messages import (
    create_success_notification_message,
    VM_STATE_STARTED,
)


logger = logging.getLogger(__name__)


class VmHealthCheckAgent(BackgroundAgent):
    def __init__(self, session_id: str):
        super().__init__(
            session_id=session_id,
            agent_type="vm_health_check_agent",
            interval=30,
        )
        self.subscribe_to(VM_HEALTH_CHECK)
        self.subscribe_to(VM_STATE_STARTED)
        self.subscribe_to(VM_DISABLE_HEALTH_CHECK_TEST)
        self._vm_datasource = VirtualMachineDataSource.get_instance()
        self._health_check_datasource = HealthCheckDataSource()
        self._notifications_service = ServiceRegistry.get(
            session_id, NOTIFICATION_SERVICE_NAME, NotificationService
        )
        self._logger = ServiceRegistry.get(session_id, LOGGER_SERVICE_NAME, LogService)
        self._time_delta_checks = timedelta(minutes=5)

    @property
    def session_id(self) -> str:
        """Get the session ID for this agent"""
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Set the session ID for this agent"""
        self._session_id = value

    async def process(self) -> None:
        """Periodic check of VM states"""
        try:
            vms = self._vm_datasource.get_vms_by_state("running")
            for vm in vms:
                logger.info(f"Checking health of VM1 {vm.name}")
                await self._process_health_check(vm.id)
        except Exception as e:
            logger.error(f"Error in VM monitor periodic check: {e}")

    async def process_message(self, message: BackgroundMessage) -> None:
        """Handle VM state change events"""
        try:
            if (
                message.message_type == VM_HEALTH_CHECK
                or message.message_type == VM_STATE_STARTED
            ):
                vm_id = message.data.get("vm_id")
                if vm_id:
                    await self._process_health_check(vm_id)
            elif message.message_type == VM_DISABLE_HEALTH_CHECK_TEST:
                vm_id = message.data.get("vm_id")
                test_name = message.data.get("test_name")
                if vm_id and test_name:
                    await self._process_disable_health_check_test(vm_id, test_name)
        except Exception as e:
            logger.error(f"Error processing security checks: {e}")

    async def _process_disable_health_check_test(
        self, vm_id: str, test_name: str
    ) -> None:
        vm = self._vm_datasource.get_vm(vm_id)
        if not vm:
            logger.error(f"VM {vm_id} not found")
            return
        self._health_check_datasource.disable_health_check_test(
            vm_id=vm_id, test_name=test_name
        )
        msg = create_success_notification_message(
            session_id=self.session_id,
            channel=vm_id,
            message=f"Health check test {test_name} disabled",
            details=f"The health check test {test_name} has been disabled for VM {vm.name}",
        )
        await self._notifications_service.send(msg)

    async def _process_health_check(self, vm_id: str) -> None:
        if not vm_id:
            logger.error("VM ID is not set")
            return
        vm = self._vm_datasource.get_vm(vm_id)
        if vm and vm.state == "running":
            logger.info(f"Checking health of VM2 {vm.name}")
            health_check = self._health_check_datasource.get_health_check(vm_id)
            if not health_check:
                health_check = VmHealthCheck(vm_id=vm_id, last_update=datetime.now())
                health_check.register_default_tests(self.session_id, vm)
                self._health_check_datasource.update_health_check(
                    vm_id=vm_id, health_check=health_check
                )
            await health_check.run_tests()
            if not health_check.is_healthy():
                logger.error(f"VM {vm_id} is not healthy: {health_check.get_reason()}")
                return
            self._health_check_datasource.update_health_check(
                vm_id=vm_id, health_check=health_check
            )
