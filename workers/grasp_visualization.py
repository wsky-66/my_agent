"""Persistent RViz point cloud, grasp geometry and selected/fallback pose markers."""
import numpy as np
from scipy.spatial.transform import Rotation
from core.grasp_geometry import visual_vertices


def grasp_markers(grasps, frame, *, selected=False):
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    result = MarkerArray()
    clear = Marker(); clear.action = Marker.DELETEALL
    result.markers.append(clear)
    edges = [(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)]
    for index, g in enumerate(grasps):
        r = np.asarray(g.get('rotation', Rotation.from_euler('xyz',g.get('rpy',[0,0,0]),degrees=True).as_matrix()))
        p = np.asarray(g['position'],dtype=float)
        color = (0.,.6,1.) if g.get('mode')=='fallback' else ((1.,.7,0.) if selected else (0.,.85,.3))
        def marker(kind, part):
            m = Marker(); m.header.frame_id = frame
            m.ns, m.id, m.type, m.action = part, index, kind, Marker.ADD
            m.pose.orientation.w = 1.
            m.color.r, m.color.g, m.color.b, m.color.a = *color, .95
            return m
        def point(xyz):
            return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        if g.get('mode')!='fallback':
            row = np.zeros(17); row[1],row[3] = g.get('width',.04),g.get('depth',.02)
            row[4:13],row[13:16] = r.ravel(),p
            boxes = visual_vertices(row[None])[0]
            wire = marker(Marker.LINE_LIST,'gripper'); wire.scale.x = .002
            wire.points = [point(box[i]) for box in boxes for a,b in edges for i in (a,b)]
            result.markers.append(wire)
        arrow = marker(Marker.ARROW,'approach'); arrow.scale.x,arrow.scale.y,arrow.scale.z = .004,.009,.012
        arrow.points = [point(p-r[:,0]*.10),point(p)]
        result.markers.append(arrow)
        label = marker(Marker.TEXT_VIEW_FACING,'label'); label.scale.z = .018
        label.pose.position = point(p+[0,0,.035])
        label.text = f"{g.get('label','object')}  " + ('fallback' if g.get('mode')=='fallback' else f"#{g.get('rank',index)} {g.get('score',0):.3f}")
        result.markers.append(label)
    return result


class GraspVisualizer:
    def __init__(self, node=None):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from visualization_msgs.msg import MarkerArray
        from sensor_msgs.msg import PointCloud2
        self.owns_context = node is None and not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.owns_node = node is None
        self.node = node or rclpy.create_node('my_agent_grasp_visualization')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cloud = self.node.create_publisher(PointCloud2,'/graspnet/pointcloud',qos) if self.owns_node else None
        self.candidates = self.node.create_publisher(MarkerArray,'/graspnet/grasps',qos) if self.owns_node else None
        self.selected = self.node.create_publisher(MarkerArray,'/graspnet/selected_grasp',qos) if not self.owns_node else None

    def publish_scene(self, points, colors, grasps, frame):
        from sensor_msgs.msg import PointCloud2, PointField
        # Preserve small segmented objects; bound only large visualization clouds.
        stride = max(1, (len(points)+29999)//30000)
        points = np.asarray(points,dtype='<f4')[::stride]
        colors = np.clip(np.asarray(colors)[::stride]*255,0,255).astype(np.uint32)
        packed = colors[:,0]<<16 | colors[:,1]<<8 | colors[:,2]
        data = np.empty((len(points),4),dtype='<f4'); data[:,:3] = points
        data[:,3] = packed.astype('<u4').view('<f4')
        msg = PointCloud2(); msg.header.frame_id = frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.height,msg.width = 1,len(points)
        msg.fields = [PointField(name=n,offset=i*4,datatype=PointField.FLOAT32,count=1)
                      for i,n in enumerate(('x','y','z','rgb'))]
        msg.point_step,msg.row_step = 16,len(points)*16
        msg.is_dense,msg.is_bigendian,msg.data = True,False,data.tobytes()
        self.cloud.publish(msg)
        self.candidates.publish(grasp_markers(grasps,frame))

    def publish_selection(self, grasp, frame):
        self.selected.publish(grasp_markers([grasp],frame,selected=True))

    def close(self):
        import rclpy
        for publisher in (self.cloud,self.candidates,self.selected):
            if publisher is not None:
                self.node.destroy_publisher(publisher)
        if self.owns_node:
            self.node.destroy_node()
        if self.owns_context:
            rclpy.shutdown()
