#! /usr/bin/env python3

import rospy
from moveit_commander import PlanningSceneInterface
from geometry_msgs.msg import Pose, PoseStamped

class CollisionManager:
    r"""
    A class to manage the collision objects in the scene.
    """


    def __init__(self, scene : PlanningSceneInterface) -> None:
        
        self._scene = scene
        self._object_counter = {
            'bottle': 0,
            'crate': 0,
            'cube': 0,
        }

    def add_abitrary_collision_object(self, pose : Pose, obj_ID : str, frame_id : str, size) -> tuple:

        obj_pose = PoseStamped()
        obj_pose.header.frame_id = frame_id
        obj_pose.pose = pose

        self._scene.add_box(obj_ID, obj_pose, size=size)

        return self.wait_for_obj_state(obj_name=obj_ID, obj_is_known=True, obj_is_attached=False), obj_ID

    def add_collision_object(self, pose : Pose, object_type : str, frame_id : str) -> tuple:

        if object_type not in self._object_counter:
            raise ValueError(f"Invalid object name: {object_type}")
        
        self._object_counter[object_type] += 1
        obj_id = f"{object_type}_{self._object_counter[object_type]:02d}"
        obj_pose = PoseStamped()
        obj_pose.header.frame_id = frame_id
        obj_pose.pose = pose

        mesh_path = f"package://ur3e_controller/meshes/{object_type}.stl"
        self._scene.add_mesh(obj_id, obj_pose, filename=mesh_path)

        return self.wait_for_obj_state(obj_name=obj_id, obj_is_known=True), obj_id
    
    def add_box_collision_object(self, pose : Pose, object_type : str, frame_id : str, size : tuple) -> tuple:

        if object_type not in self._object_counter:
            raise ValueError(f"Invalid object name: {object_type}")
        
        self._object_counter[object_type] += 1
        obj_id = f"{object_type}_{self._object_counter[object_type]:02d}"
        obj_pose = PoseStamped()
        obj_pose.header.frame_id = frame_id
        obj_pose.pose = pose

        self._scene.add_box(obj_id, obj_pose, size=size)

        return self.wait_for_obj_state(obj_name=obj_id, obj_is_known=True, obj_is_attached=True), obj_id
    
    def remove_collision_object(self, obj_id = None) -> tuple:

        if obj_id is None:
            self._scene.remove_world_object()
        else:
            self._scene.remove_world_object(obj_id)
    
    def attach_object(self, eef_link, obj_id, ):


        self._scene.attach_mesh(eef_link, obj_id, )
        return self.wait_for_obj_state(obj_name=obj_id, obj_is_known=True, obj_is_attached=True)
    
    def detach_object(self, link=None, name=None):

        self._scene.remove_attached_object(link=link, name=name)
        if name is not None:
            return self.wait_for_obj_state(obj_name=name, obj_is_known=True, obj_is_attached=False)
        else:
            return True
        
    def wait_for_obj_state(self, obj_name : str, obj_is_known : bool, obj_is_attached : bool, timeout : float = 4) -> bool:

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():

            attached_objects = self._scene.get_attached_objects([obj_name])
            is_attached = len(attached_objects.keys()) > 0

            is_known = obj_name in self._scene.get_known_object_names()

            if (obj_is_attached == is_attached) and (obj_is_known == is_known):
                return True

            rospy.sleep(0.1)
            seconds = rospy.get_time()

        return False