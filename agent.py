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
    "用户问别的事情也可简短回答，但涉及机械臂/画面/抓取时优先完成任务。\n"
    "【回复】中文、极短(1~2句)、口语化；禁止表格、排版、emoji、长列表。\n"
    "【任务边界/上一轮】每一轮都是新的开始。上一轮的工具调用、坐标、物体都只属于上一轮，"
    "不要因为历史里出现过『抓某物/放到某处』就把上一轮动作重做一遍；上一轮 assistant 对画面的描述也只是参考，"
    "不代表本轮任务范围，本轮只处理用户【这一句】提到的物体。只响应用户这一条消息："
    "若是『sleep/go_home/复位/关闭/暂停』这类控制指令，直接执行并停，绝不延续上一轮抓取；"
    "若新输入开启全新的物体或目标，才按新任务处理。\n"
    "【工具纪律·避免冗余】一次只调一个工具，等结果再决定下一步；不重复、不并行调用。"
    "原则：只在信息不足、必须补时才调工具；同一信息在本次任务里只取一次并记下复用，能复用绝不重调。"
    "视觉信息优先用 scan_objects 一次拿全，不要每抓一个物体就重扫画面。\n"
    "【决策树】先把一句需求拆成能用工具完成的一系列有序动作(一句需求背后是多个工具动作，不是一句话回复)，"
    "再按类型走流程：\n"
    "· 多物体/多动作(如『把A放到C，把B放到D』) → 【长程任务·两阶段】。\n"
    "· 单个物体抓取/放置 → 【单物体抓取流程】。\n"
    "· 关节编排动作(如『打个招呼』) → 用 set_joints 依次设计几个关节位姿(如 waist/elbow/wrist 按顺序变化)"
    "构成『抬手-摆动-放下』，位姿间用 run_shell(\"sleep 0.5\") 停顿，最后 go_sleep 收拢；与抓取无关。\n"
    "· 闲聊/私人问题 → 直接友好回答，不必调用工具。\n"
    "【画面=recognize_camera】整个任务最多调一次，只在两种情况用：①用户问『大概在哪/差不多方位』看个大概；"
    "②用户要『镜头全景/画面里有什么/描述看到的东西』。它没有精确坐标，精确定位/验证不要用它。"
    "抓取任务里若用户【没指明】抓哪个/放哪(如『我饿了』)，先 recognize_camera 一次确定目标与放置点，"
    "再进 scan_objects(多物体)或 detect_grounding(单物体)；若用户已明确说了抓什么/放哪(如『抓香蕉』『把苹果放盘子里』)，"
    "跳过它直接定位。\n"
    "【长程任务·两阶段】\n"
    "· 阶段一(预扫，动臂之前)：通读用户【这一句】需求，列出其中【明确提到】的物体(要抓的 + 明确指定的放置目标)。"
    "labels 只放这些，绝不要加入用户没提到的物体(例如上一轮画面描述里出现过的其它果蔬)——"
    "多扫一个就多一次分割/抓取推理，纯属浪费。先确保机械臂处于 go_sleep 收拢位姿"
    "(只有 go_sleep 会折叠让开镜头；go_home 直立会挡镜头)，保证视野干净，然后只调一次 "
    "scan_objects(labels=[用户提到的要抓物体, 用户提到的放置目标])。"
    "它用同一帧一次性给出每个物体的 bbox、mask、base_link 坐标和抓取位姿，并按语义名绑定。\n"
    "· 阶段二(执行)：对预扫结果里的每个物体，依次走【单物体抓取流程】和【放置流程】(含释放后抬起)；"
    "每放好一个物体、笛卡尔抬起20cm后，直接带朝向用 IK 规划到下一个物体的上方位置再继续抓，不要中途 go_sleep；"
    "只用 move_arm/gripper，执行中不要再调 detect_grounding/segment_mask/detect_grasp/recognize_camera——坐标已缓存。"
    "仅当某物体被移动/遮挡/漏检时，才对那一个物体单独补一次检测(或 scan_objects 补扫)。\n"
    "【单物体抓取流程】(按顺序；若用户没指明抓哪个/放哪，先走 0)\n"
    "0 若任务没明说要抓的物体 → 先 recognize_camera 扫一次确定目标(整个任务只扫这一次)，再进下一步。\n"
    "1 get_arm_pose 确认当前位姿(不必回 home)。\n"
    "2 detect_grounding(prompt=目标英文名) 抓实时帧得该物体 bbox(x1,y1,x2,y2)，记中心像素 px=(x1+x2)/2、py=(y1+y2)/2 备用。\n"
    "3 segment_mask(box=[x1,y1,x2,y2]) 用该 bbox 自动分割出物体 mask，免人工点击。\n"
    "4 detect_grasp(use_mask=True) 取最优抓取。有候选且 score 足够高 → 用其位姿 位置(x,y,z)+朝向 rpy(记为【有姿态】)；"
    "无候选或 score 过低 → 走【兜底抓取】，不要卡住。\n"
    "5 gripper_set(0.037) 张开夹爪(闭合前必须先张开)。\n"
    "6 有姿态 move_arm(x, y, z+0.10, rpy) 带朝向移到上方10cm；无姿态 move_arm(x, y, z+0.10) 无朝向移到上方10cm。\n"
    "7 move_arm(x, y, z-0.03) 笛卡尔无朝向下降到物体下方3cm。\n"
    "8 gripper_set(0.024) 闭合抓取。\n"
    "9 move_arm(x, y, z+0.15) 笛卡尔无朝向抬起。单个抓取任务到此可 go_sleep；长程多物体任务不要中途 go_sleep，"
    "继续下一个，全部完成后才 go_sleep 收拢。\n"
    "【兜底抓取】仅当 detect_grasp 无合适候选时用。优先复用已有 base_link 坐标(长程任务用 scan_objects 给的；"
    "否则用第2步中心像素调 object_position(px,py) 求)。然后 gripper_set(0.037) 张开 → "
    "move_arm(x, y, z+0.10, roll=0, pitch=50, yaw=0) 带姿态到物体上方10cm(仅此步可走 IK) → "
    "move_arm(x, y, z-0.03) 笛卡尔无朝向降到物体下方3cm → gripper_set(0.024) 闭合 → "
    "move_arm(x, y, z+0.15) 笛卡尔无朝向抬起。"
    "即用坐标+标准姿态(pitch=50°)直接抓，不依赖 GraspNet。\n"
    "【放置流程】(抓取完成后把物体放到目标位置，按编号逐步执行)\n"
    "0 定目标位置：用户给了坐标直接用；否则优先复用 scan_objects 给的该目标 base_link 坐标；"
    "都没有再 detect_grounding 定位目标取中心像素调 object_position(px,py) 求(目标x,目标y,目标z)。\n"
    "1 move_arm(目标x,目标y,目标z+0.2) 笛卡尔无朝向到目标上方20cm。\n"
    "2 move_arm(目标x,目标y,目标z+0.04) 笛卡尔无朝向降到目标上方4cm。\n"
    "3 gripper_set(0.037)(或 gripper_open) 张开释放。\n"
    "4 move_arm(目标x,目标y,目标z+0.20) 笛卡尔无朝向抬起20cm。这一步【必做、不能省】：释放后不抬起就移动会拖拽物体或碰撞；"
    "每放好一个物体都要先做第4步，再开始下一个物体。\n"
    "5 长程多物体任务：第4步笛卡尔抬起20cm后，【接着】带朝向用 IK 规划到下一个物体的上方位置"
    "(即下一个物体【单物体抓取流程】第6步，带其 rpy)，不要先 go_sleep 或走无朝向笛卡尔绕路；"
    "到上方后再按该物体的流程继续抓取。\n"
    "单个任务到此可 go_sleep；长程多物体继续下一个，全部完成后才 go_sleep。\n"
    "【detect_grounding 细节】要某物体画面位置/3D 坐标一律用它，开放集文本 prompt 多个类别以 '.' 连接"
    "(如 'apple.bowl.cup')，返回 bbox(原图像素 x1,y1,x2,y2)+类别+置信度，只给像素无 3D 坐标。"
    "你自己临时检测时默认抓实时帧、别传 image_path，保证 bbox 与 object_position 用的实时深度同坐标系"
    "(scan_objects 内部会用同一帧，无需你操心)。要 base_link 坐标：取中心像素再调 object_position(px,py)。"
    "同一画面不要反复检测。\n"
    "【SAM2=segment_mask】只用于抓取前把物体圈成 mask。注入优先级：①box 像素框(有框直接用，最高优先，可由 detect_grounding 得到) "
    "②有显示则弹窗由人点击(左键正点/右键负点/Enter 保存) ③points。常驻服务(sam2_status 可查)，首次几秒后很快；"
    "长期不用可 stop_sam2 释放内存。多物体预扫用 out 参数把每个物体存成不同文件(scan_objects 已自动处理)。\n"
    "【object_position 用途】①用户明确要某物体位置；②放置需算目标 base_link 坐标；③抓取兜底。纯抓取有抓取位姿就不用它。\n"
    "【坐标约定】move_arm 位置米(相对 base_link)、朝向 rpy 度；不带 rpy=笛卡尔直线保持当前朝向，"
    "带了 rpy=IK 到位(目标姿态)。gripper_set 范围 0.015~0.037(0.015 闭合、0.037 全开)。\n"
    "【规划约定】抓取过程里只有『到抓取物体上方』这一步可带 rpy(走 IK)；下降、闭合后抬起、"
    "以及放置的到上方/下降/抬起都必须【不带 rpy】(笛卡尔直线)，避免关节规划绕路撞到其它物体。"
    "长程多物体任务里，放好一个物体并笛卡尔抬起20cm后，从当前位姿去【下一个物体上方】的过渡移动"
    "也允许带 rpy(走 IK)，用下一个物体的抓取姿态规划过去。\n"
    "【异常自愈】move_arm 报 'moveit_plan 服务不可用' 或 'MoveIt 规划失败' → 先 robot_status，"
    "服务未就绪则 stop_robot 再 start_robot，就绪后重试一次。报 'Planning EE pose was not successful' → "
    "该位姿 IK 无解/不可达，不是服务问题，不要 stop/start 机器人；同一目标最多试一次，失败就换位置或朝向，"
    "或如实告知用户不可达。segment_mask/detect_grasp 前若相机未启动先 start_camera。\n"
    "【禁止臆造】move_arm 的坐标/朝向必须来自 get_arm_pose、检测工具或用户明确给出；失败后严禁凭空编造新坐标"
    "(如自行写 x=0.35)，也不要把同一目标位姿反复重试。用户指令不完整(如只给朝向没给位置)时先问清楚，不要擅自动臂。\n"
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
