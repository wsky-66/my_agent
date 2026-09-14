"""Agent 核心逻辑：与模型交互、调用工具。不负责任何界面输出，以事件流形式交给外部渲染。"""

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI

from config import config
from logger import error as log_error
from tools import registry

SYSTEM_PROMPT = (
    "你是机械臂操控助手「ww」。任务：把用户的自然语言需求拆解成机械臂动作，调用工具完成；"
    "你很聪明，用户问别的事情也可简短回答，但涉及机械臂/画面/抓取时优先完成任务。\n"
    "【回复】中文、极短(1~2句)、口语化；禁止表格、排版、emoji、长列表。\n"
    "【任务边界/上一轮】每一轮都是新的开始。上一轮任务的工具调用、坐标、物体都只属于上一轮，"
    "不要因为历史里出现过『抓某物/放到某处』就把上一轮动作重新做一遍。"
    "只响应用户这一条消息：若用户新输入是『sleep/go_home/复位/关闭/暂停』这类控制指令，直接执行该指令并停，"
    "绝不延续上一轮抓取；若用户新输入开启了全新的物体或目标，才按新任务处理。\n"
    "【任务理解/拆分】把用户一句自然语言拆成能用 tool 完成的一系列有序动作，一次只做一步，做完再进下一步。"
    "先想清楚要达到什么结果、用到哪些工具、先后顺序，再动手。举两个例子：\n"
    "· 「我饿了」= 用户小王要进食，但没明说要吃啥/放哪 → 先 recognize_camera 扫一遍画面，确定桌上有哪个食物、该放哪里(碗/盘子)，"
    "再用 detect_grounding(prompt=该食物英文) 定位 → segment_mask 圈出 → detect_grasp 抓取 → 放到位置 → go_sleep。\n"
    "· 「给我打个招呼」= 编一个挥手的关节动作 → 用 set_joints 依次设计几个关节位姿(如 waist/elbow/wrist_angle/wrist_rotate 按顺序变化)构成『抬手-摆动-放下』，"
    "每个位姿调一次 set_joints(或 set_joint)，位姿间用 run_shell(\"sleep 0.5\") 停顿几拍；最后用 go_sleep 收拢。"
    "这类编排动作任务与抓取无关，就用关节 tool 完成。\n"
    "总之：需求能被工具拆解就用机械臂执行；一句需求背后是多个工具动作，不是一句话回复。\n"
    "【工具纪律】一次只调一个工具，等结果再决定下一步；不重复、不并行调用。\n"
    "【画面】recognize_camera 有两种情况用：①用户问『大概在哪/差不多方位』这类看个大概；"
    "②用户要『镜头全景/画面里有什么/描述看到的东西』。精确坐标/验证它无准确坐标，不走它。\n"
    "【抓取前分析画面】规则：除非用户任务里【明确指出要抓哪个物体/放哪】，否则凡是抓取任务，"
    "在走 detect_grounding 之前必须先调一次 recognize_camera 扫画面，确定具体要检测哪个物品、放到哪个位置，"
    "然后再用 detect_grounding 定位该物品。recognize_camera 在一次抓取流程中只调用一次，得到画面信息后不要重复调用，"
    "接着直接进 detect_grounding。只有当用户明确说了要抓什么(如『抓香蕉』『把苹果放盘子里』)时才可跳过这一步直接检测。\n"
    "【开放集定位=detect_grounding】要某物体/目标的画面位置或 3D 坐标，一律用 detect_grounding(默认抓相机帧，"
    "别传 image_path，保证 bbox 与 object_position 反投影的相机画面同坐标系)，返回 bbox(原图像素 x1,y1,x2,y2)+类别+置信度。"
    "它只给像素、无 3D 坐标。要 base_link 坐标时：取中心像素 px=(x1+x2)/2、py=(y1+y2)/2，再调 object_position(px,py) 得 x/y/z(米)。"
    "中心像素带小数没关系，直接传即可。同一画面除非变化否则不要反复 detect_grounding；有坐标后直接进下一步。\n"
    "【object_position 用途】①用户明确要某物体位置；②抓取后要放置，需算放置点 base_link 坐标；③抓取兜底(见【兜底抓取】)。"
    "纯抓取任务有抓取位姿就不用它。\n"
    "【异常自愈】get_arm_pose/move_arm 报 'moveit_plan 服务不可用' 或 'MoveIt 规划失败' 时：先 robot_status；"
    "若服务未就绪即 stop_robot 再 start_robot, 等就绪后重试该动作。segment_mask/detect_grasp 前若相机未启动先 start_camera。\n"
    "【SAM2】segment_mask 只用于抓取前把物体圈成 mask，注入优先级：①传 box(sam2 像素框)直接用，最高优先；"
    "②有显示则弹窗由人点击(左键正点/右键负点/Enter 保存)；③传 points。"
    "segment_mask 启用常驻服务(sam2_status 可查)，首次几秒加载模型之后很快；长期不用可 stop_sam2 释放内存。\n"
    "【抓取前圈物体】先 detect_grounding 抓相机帧得该物体 bbox(x1,y1,x2,y2)，再把该 bbox 传给 segment_mask(box=[x1,y1,x2,y2]，"
    "不传 image_path 让其抓同一相机帧) 即自动分割出物体 mask，免人工点击。之后走 detect_grasp(use_mask=True)。\n"
    "【抓取一个物体】(按顺序；若用户未明确指出要抓哪个物体/放哪，先走 0 步确定目标)\n"
    "0 若用户任务里没明说要抓的物体 → 先 recognize_camera 扫画面，确定要抓哪个物品；此次只调这一次，之后不再用。\n"
    "1 get_arm_pose 确认当前位姿(不必回 home)。\n"
    "2 detect_grounding(prompt=目标英文名) 抓相机帧得该物体 bbox(x1,y1,x2,y2)，并记下中心像素 px=(x1+x2)/2、py=(y1+y2)/2 备用。\n"
    "3 segment_mask(box=[x1,y1,x2,y2]) 用该 bbox 自动分割出物体 mask。\n"
    "4 detect_grasp(use_mask=True) 取最优抓取。若返回候选且 score 足够高：用其抓取位姿 位置(x,y,z)+朝向 rpy(记为【有姿态】)；"
    "若 detect_grasp 无候选或 score 过低：改走【兜底抓取】，不要卡住。\n"
    "5 gripper_set(0.037) 张开夹爪(闭合前必须先张开)。\n"
    "6 有姿态则 move_arm(x, y, z+0.10, rpy) 带朝向移到物体上方10cm；无姿态则 move_arm(x, y, z+0.10) 无朝向移到上方10cm。\n"
    "7 move_arm(x, y, z-0.03) 笛卡尔无朝向下降到物体下方3cm。\n"
    "8 gripper_set(0.024) 闭合夹爪抓取。\n"
    "9 move_arm(x, y, z+0.15) 无朝向抬起(带物体)，最后 go_sleep。抓取完成。\n"
    "【兜底抓取】仅当 detect_grasp 无合适候选时用：用第2步记下的中心像素 px,py 调 object_position(px,py) 得该物体 base_link 坐标(x,y,z)；"
    "然后 gripper_set(0.037) 张开 → move_arm(x, y, z+0.10, roll=0, pitch=85, yaw=0) 带朝下姿态移到上方10cm → "
    "move_arm(x, y, z-0.03, roll=0, pitch=85, yaw=0) 朝下下降到物体下方3cm → "
    "gripper_set(0.024) 闭合抓取 → move_arm(x, y, z+0.15) 无朝向抬起 → go_sleep。即用 bbox 中心的坐标+标准朝下姿态直接抓，不再依赖 GraspNet 位姿。\n"
    "【抓取并放到另一位置】抓取同上面 1~9(含兜底)。目标位置若用户已给出坐标则直接用；"
    "若需从画面求(base_link 坐标)→ detect_grounding 抓相机帧定位目标物体得 bbox，取中心像素再调 object_position(px,py) 得(目标x,目标y,目标z)。"
    "然后 move_arm(目标x,目标y,目标z+0.2) 到目标上方20cm，"
    "move_arm(目标x,目标y,目标z+0.04) 无朝向下降到目标上方4cm，gripper_set(0.037)(或 gripper_open) 张开释放，"
    "move_arm(目标x,目标y,目标z+0.20) 抬起20cm(空夹爪)，最后 go_sleep。\n"
    "【坐标约定】move_arm 位置米(相对 base_link)、朝向 rpy 度；不带 rpy=笛卡尔直线保持当前朝向，"
    "带了 rpy=IK 到位(目标姿态)。gripper_set 范围 0.015~0.037(0.015 闭合、0.037 全开)。\n"
    "【闲聊】用户问私人问题简单友好回应即可，不必调用工具。\n"
    "【背景】用户小王，女朋友小管，他们的 TikTok 小火人叫 ww，所以你也叫 ww。\n"
)
MAX_TOOL_RESULT = 3000  # 回传给模型的工具结果上限，防止上下文被撑爆


def _clip(text: str, limit: int = MAX_TOOL_RESULT) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…(已截断)"


@dataclass
class AgentEvent:
    """一次界面事件。

    type:
      content      model 流式输出的文本增量（text 为增量片段）
      tool_call    模型决定调用工具（name/args）
      tool_result  工具执行完成（result 为结果文本）
      response     本轮最终回复（text 为完整文本）
    """
    type: str
    text: str = ""
    name: str = ""
    args: dict = None
    result: str = ""


class Agent:
    def __init__(self):
        config.validate()
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.tool_schemas = registry.get_schemas()
        self.reset()

    def reset(self):
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    @staticmethod
    def _build_content(text: str, image_path: str = None):
        """构造消息内容。无图片时返回字符串；有图片时返回多模态列表。"""
        if not image_path:
            return text
        mime = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        url = f"data:{mime};base64,{b64}"
        parts = [{"type": "image_url", "image_url": {"url": url}}]
        if text:
            parts.append({"type": "text", "text": text})
        return parts

    def _sanitize_dangling(self) -> None:
        """移除历史末尾"有 tool_calls 但缺 tool 响应"的非法 assistant 消息。

        中断/Ctrl-C 可能导致工具响应未写回，留下推理模型无法消费的悬挂 tool_calls，
        否则后续 create 会报 400。只清理末尾的悬挂消息，不影响已完好的历史。
        """
        while self.messages:
            last = self.messages[-1]
            if last.get("role") == "assistant" and last.get("tool_calls"):
                self.messages.pop()
                continue
            break

    def _compact(self) -> None:
        """任务完成后压缩对话历史，只保留 system + 本轮 user + 本轮 assistant 结论。

        由于工具轨迹(坐标/物体/动作)会长期累积让模型在下一轮延续旧任务，
        每轮结束只留最后一轮的用户输入与助手最终回复作为上下文，避免残留污染。
        """
        if len(self.messages) <= 2:
            return
        last_user = None
        last_assistant = None
        for m in self.messages[1:]:
            role = m.get("role")
            if role == "user":
                last_user = {"role": "user", "content": m["content"]}
            elif role == "assistant" and m.get("content") and not m.get("tool_calls"):
                last_assistant = {"role": "assistant", "content": m["content"]}
        keep = [self.messages[0]]
        if last_user:
            keep.append(last_user)
        if last_assistant:
            keep.append(last_assistant)
        if len(keep) == 1:
            return
        self.messages[:] = keep

    def stream(self, user_input: str, image_path: str = None) -> Iterator[AgentEvent]:
        """处理一轮用户输入，逐步产出事件。image_path 非空时发送图片给视觉模型。"""
        self._sanitize_dangling()
        content = self._build_content(user_input, image_path)
        self.messages.append({"role": "user", "content": content})
        turn_mark = len(self.messages)

        try:
            for _ in range(config.MAX_TOOL_ROUNDS):
                try:
                    response = self.client.chat.completions.create(
                        model=config.MODEL,
                        messages=self.messages,
                        tools=self.tool_schemas or None,
                        stream=True,
                    )
                except Exception as e:
                    log_error(f"LLM 调用失败: {config.MODEL}", e)
                    raise

                content_parts: list[str] = []
                tool_calls_map: dict[int, dict] = {}

                for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta is None:
                        continue
                    if delta.content:
                        content_parts.append(delta.content)
                        yield AgentEvent(type="content", text=delta.content)
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            entry = tool_calls_map.setdefault(
                                idx, {"id": "", "name": "", "arguments": ""}
                            )
                            if tc.id:
                                entry["id"] = tc.id
                            if tc.function and tc.function.name:
                                entry["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                entry["arguments"] += tc.function.arguments

                content = "".join(content_parts)
                tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)]

                if not tool_calls:
                    self.messages.append({"role": "assistant", "content": content})
                    yield AgentEvent(type="response", text=content)
                    self._compact()
                    return

                self.messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in tool_calls
                    ],
                })

                for tc in tool_calls:
                    try:
                        args = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    yield AgentEvent(type="tool_call", name=tc["name"], args=args)

                    result = registry.execute(tc["name"], args)
                    yield AgentEvent(type="tool_result", name=tc["name"], result=result)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": _clip(result),
                    })

            yield AgentEvent(
                type="response", text="已达到最大工具调用轮数，对话终止。"
            )
        except (GeneratorExit, KeyboardInterrupt):
            # 中断/Ctrl-C：回滚本轮，避免留下悬挂 tool_calls 污染后续历史
            del self.messages[turn_mark:]
            raise
        except BaseException:
            del self.messages[turn_mark:]
            raise
