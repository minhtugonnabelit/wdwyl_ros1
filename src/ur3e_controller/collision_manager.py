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
        }

    def add_collision_object(self, pose : Pose, object_type : str, frame_id : str) -> tuple:
        r"""
        Add a collision object to the scene.
        @param: pose        The pose of the object
        @param: object_type The type of the object
        @param: frame_id    The frame id of the object
        @returns: tuple
        """

        if object_type not in self._object_counter:
            raise ValueError(f"Invalid object name: {object_type}")
        
        self._object_counter[object_type] += 1
        obj_id = f"{object_type}_{self._object_counter[object_type]:02d}"
        obj_pose = PoseStamped()
        obj_pose.header.frame_id = frame_id
        obj_pose.pose = pose

        mesh_path = f"package://ur3e_controller/meshes/{object_type}.stl"
        self._scene.add_mesh(obj_id, obj_pose, filename=mesh_path)

        return self.wait_for_obj_state(obj_name=obj_id, obj_is_known=True)

    def wait_for_obj_state(self, obj_name : str, obj_is_known : bool, obj_is_attached : bool, timeout : float = 4) -> bool:
        r"""
        Wait for the object to be in the desired state.
        @param: obj_name        The name of the object
        @param: obj_is_known    The object is known to the scene
        @param: obj_is_attached The object is attached to the robot
        @param: timeout         The time to wait for the object to be in the desired state
        @returns: bool
        """
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
    
    def attach_object(self, eef_link, obj_id, touch_links):
        r"""
        Attach an object to the end effector of the robot.
        @param: eef_link    The end effector link of the robot
        @param: obj_id      The name of the object to be attached
        @param: touch_links The links that can touch the object

        """

        self._scene.attach_mesh(eef_link, obj_id, touch_links=touch_links)
        return self.wait_for_obj_state(obj_name=obj_id, obj_is_known=True, obj_is_attached=True)
    
    def detach_object(self, obj_id, touch_links):
        r"""
        Detach an object from the end effector of the robot.
        @param: obj_id      The name of the object to be detached
        @param: touch_links The links that can touch the object

        """

        self._scene.remove_attached_object(obj_id, touch_links=touch_links)
        return self.wait_for_obj_state(obj_name=obj_id, obj_is_known=True, obj_is_attached=False)