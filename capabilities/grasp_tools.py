"""GraspNet 抓取检测工具：订阅相机（彩色+对齐深度）→ 推理 → 返回 base_link 系抓取候选。

支持两种调用方式：
  * 单物体：detect_grasp(mask_path=...) 每次一个 mask（原有流程）。
  * 批量：detect_grasp_batch([(label, mask_path), ...]) 一次进程内加载模型一次、抓一帧，
    对多个 mask 依次推理，结果按 label 分段返回，供 scan_objects 预扫绑定语义。
"""

import os
import re
import shlex
import subprocess

from . import camera
from tools import registry

_GRASPNET_PY = "/home/wsky/miniconda3/envs/graspnet/bin/python"
_GRASPNET_SCRIPT = "/home/wsky/interbotix_ws/controller/graspnet_client.py"
_ROS_SETUP = "/opt/ros/humble/setup.bash"
_TIMEOUT = 300  # GraspNet 推理较慢（尤其 CPU），给足时间

# 新输出格式: #N: score=X xyz=[x y z] rpy_deg=[r p y] width=W
_LINE = re.compile(
    r"#(\d+):\s*score=([-\d.eE+]+)\s*xyz=\[([^\]]*)\]\s*rpy_deg=\[([^\]]*)\]\s*width=([-\d.eE+]+)"
)
# 批量输出分段: === LABEL <name> ===
_SECTION = re.compile(r"^===\s*LABEL\s+(.+?)\s*===$")

_WORKSPACE_MASK = "/home/wsky/msk_tool/masks/current_workspace.png"  # 与 sam2_tools.segment_mask 写入一致


def _grasp_from_match(m) -> dict:
    """从正则匹配构造一条抓取记录，坐标/朝向非法时返回 None。"""
    pos = [round(float(v), 3) for v in m.group(3).split()]
    rpy = [round(float(v), 1) for v in m.group(4).split()]
    if len(pos) != 3 or len(rpy) != 3:
        return None
    return {
        "rank": int(m.group(1)),
        "score": round(float(m.group(2)), 3),
        "position": pos,
        "rpy": rpy,
        "width": round(float(m.group(5)), 3),
    }


def _parse_grasps(out: str) -> list:
    grasps = []
    for line in out.splitlines():
        m = _LINE.search(line)
        if m:
            g = _grasp_from_match(m)
            if g:
                grasps.append(g)
    return grasps


def _parse_batch(out: str, labels: list) -> dict:
    """把批量输出按 === LABEL x === 分段解析成 {label: [grasps]}。"""
    result = {lb: [] for lb in labels}
    cur = None
    for line in out.splitlines():
        m = _SECTION.match(line.strip())
        if m:
            cur = m.group(1)
            result.setdefault(cur, [])
            continue
        if cur is not None:
            gm = _LINE.search(line)
            if gm:
                g = _grasp_from_match(gm)
                if g:
                    result[cur].append(g)
    return result


def _format_grasps(grasps: list, use_mask: bool = True, header: str = None) -> str:
    area = "工作区内" if use_mask else "全画幅"
    head = header or f"GraspNet 抓取候选（Top-{len(grasps)}，base_link 系，{area}）:"
    lines = [head]
    for g in grasps:
        p, r = g["position"], g["rpy"]
        lines.append(
            f"  #{g['rank']}  score={g['score']}  位置=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})m  "
            f"朝向=({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f})°  宽度={g['width']:.3f}m"
        )
    return "\n".join(lines)


def detect_grasp_raw(top: int = 5, no_collision: bool = False,
                     score_min: float = 0.0, use_mask: bool = True,
                     mask_path: str = None):
    """单物体抓取检测，返回 (grasps, err)。grasps 为候选列表，err 为错误文本。"""
    if not camera.ready():
        return None, "相机未运行或无彩色话题，请先 start_camera 再检测抓取"
    top = max(1, min(int(top), 50))
    wp = mask_path or _WORKSPACE_MASK
    if use_mask and not os.path.isfile(wp):
        return None, (f"未找到工作区 mask：{wp}\n"
                      "请先调用 segment_mask 圈定抓取工作区，再调用 detect_grasp；"
                      "或传 mask_path 指定已有 mask，或用 use_mask=False 做全画幅检测。")
    cmd = f"source {_ROS_SETUP} && {_GRASPNET_PY} {_GRASPNET_SCRIPT} --top {top}"
    if use_mask:
        cmd += f" --mask {shlex.quote(wp)}"
    if no_collision:
        cmd += " --no-collision"
    if score_min > 0:
        cmd += f" --score-min {float(score_min)}"
    try:
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, f"GraspNet 推理超时（>{_TIMEOUT}s），相机/模型可能异常或过慢"
    if proc.returncode != 0:
        return None, f"GraspNet 出错 (rc={proc.returncode}): {(proc.stderr or proc.stdout).strip()[:300]}"

    grasps = _parse_grasps(proc.stdout)
    if not grasps:
        detail = (proc.stdout + proc.stderr).strip()[:300]
        return None, f"GraspNet 未返回抓取候选（可能未收到相机帧或工作区内无可抓取物）:\n{detail}"
    return grasps, None


def detect_grasp_batch(items: list, top: int = 3, no_collision: bool = False,
                       score_min: float = 0.0):
    """批量抓取检测：一次进程加载模型、抓一帧，对多个 mask 依次推理。

    items: [(label, mask_path), ...]；返回 ({label: [grasps]}, err)。
    """
    if not camera.ready():
        return None, "相机未运行或无彩色话题，请先 start_camera 再检测抓取"
    if not items:
        return {}, None
    top = max(1, min(int(top), 50))
    for label, mp in items:
        if not mp or not os.path.isfile(mp):
            return None, f"未找到 mask（{label}）：{mp}"

    masks_arg = ",".join(f"{label}={mp}" for label, mp in items)
    cmd = (f"source {_ROS_SETUP} && {_GRASPNET_PY} {_GRASPNET_SCRIPT} "
           f"--top {top} --masks {shlex.quote(masks_arg)}")
    if no_collision:
        cmd += " --no-collision"
    if score_min > 0:
        cmd += f" --score-min {float(score_min)}"
    try:
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, f"GraspNet 批量推理超时（>{_TIMEOUT}s），相机/模型可能异常或过慢"
    if proc.returncode != 0:
        return None, f"GraspNet 出错 (rc={proc.returncode}): {(proc.stderr or proc.stdout).strip()[:300]}"

    labels = [label for label, _ in items]
    result = _parse_batch(proc.stdout, labels)
    if not any(result.values()):
        detail = (proc.stdout + proc.stderr).strip()[:300]
        return None, f"GraspNet 未返回任何抓取候选:\n{detail}"
    return result, None


@registry.register(
    name="detect_grasp",
    description="用 GraspNet 从相机（彩色+对齐深度）推理抓取候选，返回 base_link 系下的 Top-N 抓取位姿"
               "（评分、位置 x/y/z(米)、朝向 roll/pitch/yaw(度)、宽度(米)）。"
               "必须在 segment_mask 圈定工作区之后调用（默认用当前工作区 mask 限定抓取范围）；"
               "未设置工作区会提示先 segment_mask，不会在全画幅里出候选。"
               "score_min 滤低置信框、no_collision 跳过碰撞检测。"
               "适合在抓取目标前定位可抓取点。需要相机已启动。",
)
def detect_grasp(top: int = 5, no_collision: bool = False,
                 score_min: float = 0.0, use_mask: bool = True,
                 mask_path: str = None) -> str:
    grasps, err = detect_grasp_raw(top, no_collision, score_min, use_mask, mask_path)
    if err:
        return err
    return _format_grasps(grasps, use_mask)
