from typing import Tuple
from abc import ABC, abstractmethod
import logging
from pd_ai_agent_core.parallels_desktop import VirtualMachine
from pd_ai_agent_core.services.service_registry import ServiceRegistry
from pd_ai_agent_core.services.notification_service import NotificationService

from pd_ai_agent_core.common import (
    NOTIFICATION_SERVICE_NAME,
)
from pd_ai_agent_core.messages import Message

logger = logging.getLogger(__name__)

FAILURE_MESSAGE = "Health check test failed"
RECOVERY_MESSAGE = "Health check test recovered"


class VMHealthCheckTestResult:
    def __init__(self, is_healthy: bool, reason: str):
        self.is_healthy = is_healthy

        self.reason = reason


class VMHealthCheckTest(ABC):
    def __init__(
        self, session_id: str, vm: VirtualMachine, name: str, count_for_failure: int
    ):
        self.session_id = session_id
        self.vm = vm
        self.name = name
        self.count_for_failure = count_for_failure
        self._ignore_test = False
        self._count = 0
        self.reason = ""
        self.notifications_service = ServiceRegistry.get(
            session_id, NOTIFICATION_SERVICE_NAME, NotificationService
        )

    @abstractmethod
    async def _check_function(self) -> Tuple[bool, str]:
        pass

    @abstractmethod
    def _failure_message(self) -> Message:
        pass

    @abstractmethod
    def _recovery_message(self) -> Message:
        pass

    async def check(self) -> VMHealthCheckTestResult:
        is_healthy, reason = await self._check_function()
        return VMHealthCheckTestResult(is_healthy, reason)

    async def run(self) -> Tuple[bool, str]:
        is_healthy, reason = await self._check_function()
        self.reason = reason
        if is_healthy:
            if self._count > 0:
                msg = self._recovery_message()
                logger.info(f"Sending recovery message: {msg}")
                await self.notifications_service.send(msg)
            self._count = 0
            return True, ""
        self._count += 1
        if self._count >= self.count_for_failure:
            msg = self._failure_message()
            logger.info(f"Sending failure message: {msg}")
            await self.notifications_service.send(msg)
        return False, reason

    def increment_failure(self) -> None:
        self._count += 1

    def disable(self) -> None:
        self._ignore_test = True
        self._count = 0

    def is_disabled(self) -> bool:
        return self._ignore_test

    def is_healthy(self) -> bool:
        return self._count < self.count_for_failure
