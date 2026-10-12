"""Batch forward kinematics and visual sampling of a timed joint trajectory."""
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


class LinkChain:
    def __init__(self, description, base, tip):
        joints = {j.find('child').attrib['link']:j for j in ET.fromstring(description).findall('joint')}
        chain,seen = [],set()
        while tip != base:
            if tip in seen or tip not in joints:
                raise ValueError('机器人描述中没有所需的基座到末端链')
            seen.add(tip)
            joint = joints[tip]
            kind = joint.attrib['type']
            if kind not in {'fixed','revolute','continuous','prismatic'}:
                raise ValueError('末端链含不支持的关节类型')
            origin = np.eye(4)
            element = joint.find('origin')
            if element is not None:
                origin[:3,3] = np.fromstring(element.get('xyz','0 0 0'),sep=' ')
                origin[:3,:3] = Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
            axis = np.array([1.,0.,0.])
            if joint.find('axis') is not None:
                axis = np.fromstring(joint.find('axis').get('xyz'),sep=' ')
            if kind != 'fixed':axis = axis/np.linalg.norm(axis)
            if not np.isfinite(origin).all() or not np.isfinite(axis).all():
                raise ValueError('机器人描述中的关节变换无效')
            chain.append((joint.attrib['name'],kind,origin,axis))
            tip = joint.find('parent').attrib['link']
        self.chain = chain[::-1]

    def positions(self, names, samples):
        indexes = {name:i for i,name in enumerate(names)}
        result = np.broadcast_to(np.eye(4),(len(samples),4,4)).copy()
        for name,kind,origin,axis in self.chain:
            result = result @ origin
            if kind == 'fixed':continue
            if name not in indexes:
                raise ValueError(f'规划轨迹缺少末端链关节 {name}')
            values = samples[:,indexes[name]]
            motion = np.broadcast_to(np.eye(4),result.shape).copy()
            if kind == 'prismatic':motion[:,:3,3] = values[:,None]*axis
            else:motion[:,:3,:3] = Rotation.from_rotvec(values[:,None]*axis).as_matrix()
            result = result @ motion
        return result[:,:3,3]


def sample_trajectory(trajectory, time_step=.02, angle_step=.01):
    """Use the controller's linear/cubic/quintic rule without modifying its goal."""
    count = len(trajectory.joint_names)
    result = [np.asarray(trajectory.points[0].positions,dtype=float)[None]]
    for a,b in zip(trajectory.points,trajectory.points[1:]):
        stamp = lambda p:p.time_from_start.sec+p.time_from_start.nanosec*1e-9
        duration = stamp(b)-stamp(a)
        if duration<=0:raise ValueError('轨迹采样时间必须递增')
        start,end = np.asarray(a.positions),np.asarray(b.positions)
        steps = max(1,math.ceil(duration/time_step),math.ceil(float(np.max(np.abs(end-start)))/angle_step))
        t = (np.arange(1,steps+1,dtype=float)/steps)[:,None]
        have_velocity = all(len(getattr(p,'velocities',()))==count for p in (a,b))
        have_acceleration = have_velocity and all(len(getattr(p,'accelerations',()))==count for p in (a,b))
        if have_acceleration:
            c1 = np.asarray(a.velocities)*duration
            c2 = np.asarray(a.accelerations)*duration**2/2
            d = end-start-c1-c2
            v = np.asarray(b.velocities)*duration-c1-2*c2
            acc = np.asarray(b.accelerations)*duration**2-2*c2
            c3,c4,c5 = 10*d-4*v+acc/2, -15*d+7*v-acc, 6*d-3*v+acc/2
            values = start+t*(c1+t*(c2+t*(c3+t*(c4+t*c5))))
        elif have_velocity:
            v0,v1 = np.asarray(a.velocities)*duration,np.asarray(b.velocities)*duration
            d = end-start
            values = start+t*(v0+t*(3*d-2*v0-v1+t*(-2*d+v0+v1)))
        else:
            values = start+(end-start)*t
        values[-1] = end
        result.append(values)
    samples = np.concatenate(result)
    if samples.shape[1]!=count or not np.isfinite(samples).all():
        raise ValueError('轨迹采样关节数据无效')
    return samples
