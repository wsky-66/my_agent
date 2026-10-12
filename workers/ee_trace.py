"""Opaque end-effector path points, computed locally from the live robot URDF."""
import threading

from core.kinematics import LinkChain, sample_trajectory

TOPIC = '/my_agent/ee_trajectory'


def trace_markers(points, frame, stamp):
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray
    result = MarkerArray()
    clear = Marker();clear.action = Marker.DELETEALL
    result.markers.append(clear)
    positions = [Point(x=float(p[0]),y=float(p[1]),z=float(p[2])) for p in points]
    colors = [ColorRGBA(r=float(.1+.9*i/max(1,len(points)-1)),g=float(.8-.3*i/max(1,len(points)-1)),
                        b=float(1.-.85*i/max(1,len(points)-1)),a=1.) for i in range(len(points))]
    def marker(kind,name,size):
        m = Marker();m.header.frame_id,m.header.stamp = frame,stamp
        m.ns,m.id,m.type,m.action = name,0,kind,Marker.ADD
        m.pose.orientation.w = 1.
        m.scale.x=m.scale.y=m.scale.z=size
        m.color.a=1.
        return m
    path = marker(Marker.LINE_STRIP,'ee_path',.0012)
    path.points,path.colors = positions,colors
    dots = marker(Marker.POINTS,'ee_samples',.0035)
    dots.points,dots.colors = positions,colors
    ends = marker(Marker.SPHERE_LIST,'ee_endpoints',.008)
    ends.points = [positions[0],positions[-1]]
    ends.colors = [ColorRGBA(r=.1,g=1.,b=.4,a=1.),ColorRGBA(r=1.,g=.45,b=.1,a=1.)]
    result.markers.extend([path,dots,ends])
    return result


class EndEffectorTrace:
    def __init__(self, node, robot_name, base, tip):
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray
        from rclpy.qos import QoSProfile, DurabilityPolicy
        self.node,self.base,self.tip = node,base,tip
        self._lock = threading.RLock()
        self._chain,self._trajectory = None,None
        qos = QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = node.create_publisher(MarkerArray,TOPIC,qos)
        self.subscription = node.create_subscription(String,f'/{robot_name}/robot_description',self._description,qos)

    def _description(self, message):
        try:
            chain = LinkChain(message.data,self.base,self.tip)
            with self._lock:
                self._chain = chain
                if self._trajectory is not None:self.publish(self._trajectory)
        except Exception as exc:
            from logger import error
            error('RViz 末端轨迹机器人描述加载失败',exc)

    def publish(self, trajectory):
        # No ROS FK calls or hardware-feedback waits in the motion path. If the
        # latched model arrives later, its callback draws the latest plan then.
        with self._lock:
            self._trajectory = trajectory
            if self._chain is None:return
            try:
                samples = sample_trajectory(trajectory)
                points = self._chain.positions(trajectory.joint_names,samples)
                self.publisher.publish(trace_markers(points,self.base,self.node.get_clock().now().to_msg()))
            except Exception as exc:
                from visualization_msgs.msg import Marker, MarkerArray
                clear = Marker();clear.action=Marker.DELETEALL
                self.publisher.publish(MarkerArray(markers=[clear]))
                from logger import error
                error('RViz 末端轨迹点发布失败',exc)

    def close(self):
        self.node.destroy_subscription(self.subscription)
        self.node.destroy_publisher(self.publisher)
