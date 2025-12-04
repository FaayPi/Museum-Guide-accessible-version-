"""
Health Check and System Monitoring Module.

Provides health check endpoints for production monitoring,
load balancers, and orchestration systems (Kubernetes, Docker Swarm, etc.)
"""

import time
from typing import Dict, Any, List
from datetime import datetime
import psutil
import logging

logger = logging.getLogger(__name__)


class HealthCheck:
    """
    Comprehensive health check system.

    Monitors:
    - API connectivity (OpenAI, Pinecone)
    - System resources (CPU, Memory, Disk)
    - Dependencies status
    - Application uptime
    """

    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.

        Returns:
            Dictionary with health status information

        Status codes:
            - healthy: All systems operational
            - degraded: Some non-critical issues
            - unhealthy: Critical issues detected
        """
        status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'uptime_seconds': round(time.time() - self.start_time, 2),
            'version': '1.0.0',
            'checks': {}
        }

        # System resources check
        try:
            status['checks']['system'] = self._check_system_resources()
        except Exception as e:
            logger.error(f"System check failed: {e}")
            status['checks']['system'] = {'status': 'error', 'message': str(e)}

        # API connectivity check
        try:
            status['checks']['apis'] = self._check_api_connectivity()
        except Exception as e:
            logger.error(f"API check failed: {e}")
            status['checks']['apis'] = {'status': 'error', 'message': str(e)}

        # Dependencies check
        try:
            status['checks']['dependencies'] = self._check_dependencies()
        except Exception as e:
            logger.error(f"Dependencies check failed: {e}")
            status['checks']['dependencies'] = {'status': 'error', 'message': str(e)}

        # Request statistics
        status['statistics'] = {
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': round(self.error_count / max(self.request_count, 1), 3)
        }

        # Determine overall status
        if any(check.get('status') == 'unhealthy' for check in status['checks'].values()):
            status['status'] = 'unhealthy'
        elif any(check.get('status') == 'degraded' for check in status['checks'].values()):
            status['status'] = 'degraded'

        return status

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        status = 'healthy'
        warnings = []

        # CPU threshold: 80%
        if cpu_percent > 80:
            status = 'degraded'
            warnings.append(f"High CPU usage: {cpu_percent}%")

        # Memory threshold: 85%
        if memory.percent > 85:
            status = 'degraded'
            warnings.append(f"High memory usage: {memory.percent}%")

        # Disk threshold: 90%
        if disk.percent > 90:
            status = 'degraded'
            warnings.append(f"High disk usage: {disk.percent}%")

        return {
            'status': status,
            'cpu_percent': round(cpu_percent, 2),
            'memory_percent': round(memory.percent, 2),
            'memory_available_mb': round(memory.available / 1024 / 1024, 2),
            'disk_percent': round(disk.percent, 2),
            'disk_free_gb': round(disk.free / 1024 / 1024 / 1024, 2),
            'warnings': warnings
        }

    def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check connectivity to external APIs."""
        checks = {}

        # OpenAI API check
        try:
            import config
            if config.OPENAI_API_KEY:
                checks['openai'] = {'status': 'healthy', 'configured': True}
            else:
                checks['openai'] = {'status': 'unhealthy', 'configured': False}
        except Exception as e:
            checks['openai'] = {'status': 'error', 'message': str(e)}

        # Pinecone API check
        try:
            import config
            if config.PINECONE_API_KEY:
                checks['pinecone'] = {'status': 'healthy', 'configured': True}
            else:
                checks['pinecone'] = {'status': 'unhealthy', 'configured': False}
        except Exception as e:
            checks['pinecone'] = {'status': 'error', 'message': str(e)}

        # Overall API status
        if all(c.get('status') == 'healthy' for c in checks.values()):
            overall_status = 'healthy'
        elif any(c.get('status') == 'unhealthy' for c in checks.values()):
            overall_status = 'unhealthy'
        else:
            overall_status = 'degraded'

        return {
            'status': overall_status,
            'details': checks
        }

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies."""
        missing_deps = []
        available_deps = []

        required_packages = [
            'openai',
            'pinecone',
            'gradio',
            'PIL',
            'imagehash',
            'numpy'
        ]

        for package in required_packages:
            try:
                __import__(package)
                available_deps.append(package)
            except ImportError:
                missing_deps.append(package)

        status = 'healthy' if not missing_deps else 'unhealthy'

        return {
            'status': status,
            'available': available_deps,
            'missing': missing_deps,
            'total_required': len(required_packages)
        }

    def increment_request(self):
        """Increment request counter."""
        self.request_count += 1

    def increment_error(self):
        """Increment error counter."""
        self.error_count += 1

    def get_liveness(self) -> Dict[str, str]:
        """
        Simple liveness probe for Kubernetes.

        Returns:
            Dict indicating application is running
        """
        return {
            'status': 'alive',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }

    def get_readiness(self) -> Dict[str, Any]:
        """
        Readiness probe for Kubernetes.

        Returns:
            Dict indicating application is ready to serve traffic
        """
        status = self.get_status()

        # Application is ready if not unhealthy
        is_ready = status['status'] != 'unhealthy'

        return {
            'status': 'ready' if is_ready else 'not_ready',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'details': status
        }


# Global health check instance
_health_check = None


def get_health_check() -> HealthCheck:
    """Get or create global health check instance."""
    global _health_check
    if _health_check is None:
        _health_check = HealthCheck()
    return _health_check


# Convenience functions
def health_status() -> Dict[str, Any]:
    """Get current health status."""
    return get_health_check().get_status()


def liveness_probe() -> Dict[str, str]:
    """Liveness probe endpoint."""
    return get_health_check().get_liveness()


def readiness_probe() -> Dict[str, Any]:
    """Readiness probe endpoint."""
    return get_health_check().get_readiness()
