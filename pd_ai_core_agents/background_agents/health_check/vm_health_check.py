from datetime import datetime
from typing import List
from .vm_health_check_test import (
    VMHealthCheckTest,
)
from .health_checks.detect_black_screen import (
    DetectBlackScreenHealthCheckTest,
)
from .health_checks.detect_guest_tools import (
    DetectGuestToolsHealthCheckTest,
)
from .health_checks.detect_internet_connection import (
    DetectInternetConnectionHealthCheckTest,
)
from pd_ai_agent_core.parallels_desktop import VirtualMachine
import logging

logger = logging.getLogger(__name__)


class VmHealthCheck:
    def __init__(self, vm_id: str, last_update: datetime):
        self.vm_id = vm_id
        self.last_update = last_update
        self._is_healthy = True
        self._reason = ""
        self.tests: List[VMHealthCheckTest] = []

    def register_test(self, test: VMHealthCheckTest) -> None:
        self.tests.append(test)

    def register_default_tests(self, session_id: str, vm: VirtualMachine) -> None:
        self.register_test(
            DetectBlackScreenHealthCheckTest(session_id=session_id, vm=vm)
        )
        self.register_test(
            DetectInternetConnectionHealthCheckTest(session_id=session_id, vm=vm)
        )
        self.register_test(
            DetectGuestToolsHealthCheckTest(session_id=session_id, vm=vm)
        )

    async def run_tests(self) -> None:
        for test in self.tests:
            if test.is_disabled():
                continue
            logger.info(f"Running test {test.name}")
            result, _ = await test.run()
            logger.info(f"Test {test.name} finished with result {result}")

    def is_healthy(self) -> bool:
        for test in self.tests:
            if not test.is_healthy():
                self._is_healthy = False
                self._reason = test.reason
                return False
        return True

    def get_reason(self) -> str:
        if not self._is_healthy:
            return self._reason
        return "Healthy"

    def get_tests(self) -> List[VMHealthCheckTest]:
        return self.tests

    def disable_test(self, test_name: str) -> None:
        for test in self.tests:
            if test.name.lower() == test_name.lower():
                test.disable()
                return
        raise ValueError(f"Test {test_name} not found")
