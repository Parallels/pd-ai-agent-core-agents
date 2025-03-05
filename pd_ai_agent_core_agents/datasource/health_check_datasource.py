from typing import Dict, Optional
import logging
from datetime import datetime, timedelta
from background_agents.health_check_agent.vm_health_check import VmHealthCheck

logger = logging.getLogger(__name__)


class HealthCheckDataSource:
    def __init__(self):
        self._vms: Dict[str, VmHealthCheck] = {}
        self._last_update: Optional[datetime] = None
        self._cache_duration: timedelta = timedelta(minutes=10)

    def update_health_check(self, vm_id: str, health_check: VmHealthCheck) -> None:
        """Update the last update time for a VM"""
        self._vms[vm_id] = health_check

    def disable_health_check_test(self, vm_id: str, test_name: str) -> None:
        """Disable a health check test for a VM"""
        if vm_id in self._vms:
            self._vms[vm_id].disable_test(test_name)

    def remove_health_check(self, vm_id: str) -> None:
        """Remove a health check for a VM"""
        if vm_id in self._vms:
            del self._vms[vm_id]

    def get_health_check(self, vm_id: str) -> Optional[VmHealthCheck]:
        """Get the health check for a VM"""
        return self._vms.get(vm_id)

    def get_last_check(self, vm_id: str) -> Optional[VmHealthCheck]:
        """Get the last update time for a VM"""
        return self._vms.get(vm_id)
        return self._vms.get(vm_id)

    def is_cache_valid(self) -> bool:
        """Check if the cache is still valid"""
        if self._last_update is None:
            return False
        return datetime.now() - self._last_update < self._cache_duration

    def clear_cache(self) -> None:
        """Clear the cache"""
        self._vms.clear()
        self._last_update = None
