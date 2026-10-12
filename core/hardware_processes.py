"""Linux hardware process inventory, exact device scope and PID identity checks."""
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import signal
import time

from core.results import ToolResult


class SessionLock:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise RuntimeError("另一实例正在使用本项目硬件，请先退出该实例")
        self.file.seek(0)
        self.file.truncate()
        self.file.write(str(os.getpid()))
        self.file.flush()

    def close(self):
        if not self.file.closed:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    uid: int
    started: int
    state: str
    args: tuple
    cwd: str = ""
    domain: str = "0"


def read_process(pid, proc_root=Path("/proc")):
    path = Path(proc_root) / str(pid)
    try:
        fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
        args = tuple(p.decode(errors="replace") for p in (path / "cmdline").read_bytes().split(b"\0") if p)
        try:
            environment = (path / "environ").read_bytes().split(b"\0")
            domain = next((v.split(b"=", 1)[1].decode() for v in environment if v.startswith(b"ROS_DOMAIN_ID=")), "0") or "0"
        except OSError:
            domain = "0"
        return ProcessRecord(int(pid), int(fields[1]), path.stat().st_uid, int(fields[19]),
                             fields[0], args, str((path / "cwd").resolve()), domain)
    except (OSError, ValueError, IndexError):
        return None


def launch_options(args):
    return dict(arg.split(":=", 1) for arg in args if ":=" in arg)


def is_ros_launch(args):
    return bool(args and (Path(args[0]).name == "ros2" or
                len(args) > 1 and Path(args[0]).name.startswith("python") and Path(args[1]).name == "ros2"))


def related(process, robot_name, camera_prefix, project_root, *, arm=True, camera=True, domain="0"):
    args = process.args
    if not args or process.domain != domain:
        return False
    if "-m" in args and process.cwd == str(project_root):
        module = args[args.index("-m")+1] if args.index("-m")+1 < len(args) else ""
        if module.startswith("workers.") and module.endswith("_worker"):
            return True
    options = launch_options(args)
    if is_ros_launch(args) and "launch" in args:
        index = args.index("launch")
        package = args[index+1] if index+1 < len(args) else ""
        filename = args[index+2] if index+2 < len(args) else ""
        launches = {("interbotix_xsarm_moveit_interface", "xsarm_moveit_interface.launch.py"),
                    ("interbotix_xsarm_moveit", "xsarm_moveit.launch.py"),
                    ("interbotix_xsarm_ros_control", "xsarm_ros_control.launch.py"),
                    ("interbotix_xsarm_control", "xsarm_control.launch.py")}
        if arm and (package, filename) in launches and options.get("robot_name", options.get("robot_model")) == robot_name:
            return True
        prefix = "/" + "/".join(p.strip("/") for p in (
            options.get("camera_namespace", "camera"), options.get("camera_name", "camera")) if p.strip("/"))
        if camera and (package, filename) == ("realsense2_camera", "rs_launch.py") and prefix == camera_prefix:
            return True
    executable = Path(args[0]).name
    if executable.startswith("python") and len(args) > 1:
        executable = Path(args[1]).name
    robot_nodes = {"xs_sdk", "xs_sdk_sim.py", "ros2_control_node", "spawner", "move_group",
                   "moveit_interface", "moveit_interface_gui", "robot_state_publisher", "rviz2"}
    namespace = options.get("__ns", "").rstrip("/")
    if arm and executable in robot_nodes and (namespace == "/"+robot_name or
        any(":=/"+robot_name+"/" in arg or arg.startswith("/"+robot_name+"/") for arg in args)):
        return True
    camera_namespace, _, camera_name = camera_prefix.rpartition("/")
    if camera and executable == "realsense2_camera_node" and namespace == camera_namespace and options.get("__node", "camera") == camera_name:
        return True
    return False


class HardwareSweep:
    def __init__(self, robot_name, camera_prefix, project_root, *, arm=True, camera=True, serial_port=None,
                 proc_root="/proc", sys_root="/sys", uid=None, protected=None):
        self.robot_name, self.camera_prefix = robot_name, camera_prefix
        self.project_root = str(Path(project_root).resolve())
        self.arm, self.camera = arm, camera
        self.proc_root, self.sys_root = Path(proc_root), Path(sys_root)
        self.uid = os.getuid() if uid is None else uid
        self.domain = os.environ.get("ROS_DOMAIN_ID", "0") or "0"
        self.protected = set(protected or ()) | {os.getpid()}
        parent = read_process(os.getpid(), self.proc_root)
        while parent and parent.ppid not in self.protected:
            self.protected.add(parent.ppid)
            parent = read_process(parent.ppid, self.proc_root)
        self.devices = set()
        if arm and serial_port and Path(serial_port).exists():
            self.devices.add(str(Path(serial_port).resolve()))
        if camera:
            cameras = {}
            for path in (self.sys_root / "class/video4linux").glob("video*"):
                try:
                    if "realsense" in (path / "name").read_text().lower():
                        # Several video nodes can belong to one physical USB camera.
                        physical = (path / "device").resolve().parent
                        cameras.setdefault(physical, set()).add("/dev/"+path.name)
                except OSError:
                    pass
            if len(cameras) == 1:
                physical, videos = next(iter(cameras.items()))
                self.devices.update(videos)
                try:
                    bus = int((physical / "busnum").read_text())
                    device = int((physical / "devnum").read_text())
                    self.devices.add(f"/dev/bus/usb/{bus:03d}/{device:03d}")
                except (OSError, ValueError):
                    pass

    def snapshot(self):
        records = {}
        for path in self.proc_root.iterdir():
            if path.name.isdigit():
                process = read_process(path.name, self.proc_root)
                if process and process.uid == self.uid and process.pid not in self.protected and process.state != "Z":
                    records[process.pid] = process
        return records

    def owns_device(self, process):
        for path in (self.proc_root / str(process.pid) / "fd").glob("*"):
            try:
                if str(path.resolve()) in self.devices:
                    return True
            except OSError:
                continue
        return False

    def targets(self, owned_pids=()):
        records = self.snapshot()
        selected = {pid: p for pid, p in records.items() if pid in owned_pids or self.owns_device(p) or
                    related(p, self.robot_name, self.camera_prefix, self.project_root,
                            arm=self.arm, camera=self.camera, domain=self.domain)}
        while True:
            children = {pid: p for pid, p in records.items() if p.ppid in selected and pid not in selected}
            if not children:
                break
            selected.update(children)
        return selected

    def device_holders(self):
        holders = {}
        if not self.devices:
            return holders
        for path in self.proc_root.iterdir():
            if path.name.isdigit():
                process = read_process(path.name, self.proc_root)
                if process and process.uid == self.uid and process.state != "Z" and self.owns_device(process):
                    holders[process.pid] = process
        return holders

    def alive(self, process):
        now = read_process(process.pid, self.proc_root)
        return bool(now and now.uid == process.uid and now.started == process.started and now.state != "Z")

    def _signal(self, process, sig):
        if process.pid in self.protected or process.uid != self.uid:
            return False
        if self.alive(process):
            try:
                if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                    fd = os.pidfd_open(process.pid)
                    try:
                        if self.alive(process):
                            signal.pidfd_send_signal(fd, sig)
                        else:
                            return False
                    finally:
                        os.close(fd)
                else:
                    os.kill(process.pid, sig)
                return True
            except ProcessLookupError:
                pass
        return False

    def clean(self, captured=None, grace=3):
        selected = dict(captured or {})
        selected.update(self.targets())
        terminated = set()
        errors = []
        for _ in range(2):
            for p in selected.values():
                try:
                    if self._signal(p, signal.SIGINT if is_ros_launch(p.args) else signal.SIGTERM):
                        terminated.add(p.pid)
                except PermissionError:
                    errors.append(p.pid)
            until = time.monotonic()+grace
            while time.monotonic() < until and any(self.alive(p) for p in selected.values()):
                time.sleep(.05)
            for p in selected.values():
                try:
                    self._signal(p, signal.SIGKILL)
                except PermissionError:
                    errors.append(p.pid)
            until = time.monotonic()+1
            while time.monotonic() < until and any(self.alive(p) for p in selected.values()):
                time.sleep(.05)
            survivors = {pid: p for pid, p in selected.items() if self.alive(p)}
            survivors.update(self.targets())
            # Protected ancestors must not be killed, but their device handles
            # still mean we cannot claim that the hardware has been released.
            survivors.update(self.device_holders())
            if not survivors:
                break
            selected = survivors
        data = {"terminated_pids": sorted(terminated), "remaining_pids": sorted(survivors),
                "permission_denied_pids": sorted(set(errors)), "devices": sorted(self.devices),
                "protected_holders": sorted(set(survivors) & self.protected)}
        if survivors:
            return ToolResult.failure("HARDWARE_STILL_BUSY", "仍有硬件进程或设备占用，不能确认清理完成", data)
        return ToolResult.success("相关进程已退出，未发现残留设备占用", data)
