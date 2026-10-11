"""Read-only arm health isolated in the configured ROS Python environment."""
from config import config
from services.runtime import worker

client = worker("health", config.ROS_PYTHON, {"robot_name": config.ROBOT_NAME},
                ros=True, workspace=True, timeout=5)
client.startup_timeout = 10


def robot_status(*, allow_shutdown=False):
    result = client.request("robot", allow_shutdown=allow_shutdown)
    if not result.ok and result.error_code not in ("TASK_CANCELLED", "WORKER_UNAVAILABLE", "WORKER_TIMEOUT"):
        serial = client.request("serial", allow_shutdown=allow_shutdown)
        if not serial.ok:
            serial.data = {**serial.data, "checks": result.data.get("checks", {})}
            return serial
    return result


def serial_preflight(*, allow_shutdown=False):
    return client.request("serial", allow_shutdown=allow_shutdown)


def stop():
    client.stop()
