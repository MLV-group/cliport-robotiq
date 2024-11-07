"""Classes to handle gripper dynamics."""

import os

import numpy as np
from cliport.utils import pybullet_utils
import time
import pybullet as p

SPATULA_BASE_URDF = 'ur5/spatula/spatula-base.urdf'
SUCTION_BASE_URDF = 'ur5/suction/suction-base.urdf'
SUCTION_HEAD_URDF = 'ur5/suction/suction-head.urdf'
GRIPPER_URDF = 'ur5/gripper/robotiq_2f_85.urdf'
UR5_URDF_PATH = 'ur5/ur5.urdf'

class Gripper:
    """Base gripper class."""

    def __init__(self, assets_root):
        self.assets_root = assets_root
        self.activated = False

    def step(self):
        """This function can be used to create gripper-specific behaviors."""
        return

    def activate(self, objects):
        del objects
        return

    def release(self):
        return


class Spatula(Gripper):
    """Simulate simple spatula for pushing."""

    def __init__(self, assets_root, robot, ee, obj_ids):  # pylint: disable=unused-argument
        """Creates spatula and 'attaches' it to the robot."""
        super().__init__(assets_root)

        # Load spatula model.
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, SPATULA_BASE_URDF), pose[0], pose[1])
        p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))


class Suction(Gripper):
    """Simulate simple suction dynamics."""

    def __init__(self, assets_root, robot, ee, obj_ids):
        """Creates suction and 'attaches' it to the robot.
    
        Has special cases when dealing with rigid vs deformables. For rigid,
        only need to check contact_constraint for any constraint. For soft
        bodies (i.e., cloth or bags), use cloth_threshold to check distances
        from gripper body (self.body) to any vertex in the cloth mesh. We
        need correct code logic to handle gripping potentially a rigid or a
        deformable (and similarly for releasing).
    
        To be clear on terminology: 'deformable' here should be interpreted
        as a PyBullet 'softBody', which includes cloths and bags. There's
        also cables, but those are formed by connecting rigid body beads, so
        they can use standard 'rigid body' grasping code.
    
        To get the suction gripper pose, use p.getLinkState(self.body, 0),
        and not p.getBasePositionAndOrientation(self.body) as the latter is
        about z=0.03m higher and empirically seems worse.
    
        Args:
          assets_root: str for root directory with assets.
          robot: int representing PyBullet ID of robot.
          ee: int representing PyBullet ID of end effector link.
          obj_ids: list of PyBullet IDs of all suctionable objects in the env.
        """
        super().__init__(assets_root)

        # Load gripper base model (visual only).
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
        p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))

        # Load suction tip model (visual and collision) with compliance.
        # urdf = 'assets/ur5/suction/suction-head.urdf'
        pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.body = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=self.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08))
        p.changeConstraint(constraint_id, maxForce=50)

        # Reference to object IDs in environment for simulating suction.
        self.obj_ids = obj_ids

        # Indicates whether gripper is gripping anything (rigid or def).
        self.activated = False

        # For gripping and releasing rigid objects.
        self.contact_constraint = None

        # Defaults for deformable parameters, and can override in tasks.
        self.def_ignore = 0.035  # TODO(daniel) check if this is needed
        self.def_threshold = 0.030
        self.def_nb_anchors = 1

        # Track which deformable is being gripped (if any), and anchors.
        self.def_grip_item = None
        self.def_grip_anchors = []

        # Determines release when gripped deformable touches a rigid/def.
        # TODO(daniel) should check if the code uses this -- not sure?
        self.def_min_vetex = None
        self.def_min_distance = None

        # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
        self.init_grip_distance = None
        self.init_grip_item = None

    def activate(self):
        """Simulate suction using a rigid fixed constraint to contacted object."""
        # TODO(andyzeng): check deformables logic.
        # del def_ids

        if not self.activated:
            points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
            # print(points)
            if points:

                # Handle contact between suction with a rigid object.
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                if obj_id in self.obj_ids['rigid']:
                    body_pose = p.getLinkState(self.body, 0)
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                       world_to_body[1],
                                                       obj_pose[0], obj_pose[1])
                    self.contact_constraint = p.createConstraint(
                        parentBodyUniqueId=self.body,
                        parentLinkIndex=0,
                        childBodyUniqueId=obj_id,
                        childLinkIndex=contact_link,
                        jointType=p.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0))

                self.activated = True

    def release(self):
        """Release gripper object, only applied if gripper is 'activated'.
    
        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.
    
        To handle deformables, simply remove constraints (i.e., anchors).
        Also reset any relevant variables, e.g., if releasing a rigid, we
        should reset init_grip values back to None, which will be re-assigned
        in any subsequent grasps.
        """
        if self.activated:
            self.activated = False

            # Release gripped rigid object (if any).
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:  # pylint: disable=bare-except
                    pass
                self.init_grip_distance = None
                self.init_grip_item = None

            # Release gripped deformable object (if any).
            if self.def_grip_anchors:
                for anchor_id in self.def_grip_anchors:
                    p.removeConstraint(anchor_id)
                self.def_grip_anchors = []
                self.def_grip_item = None
                self.def_min_vetex = None
                self.def_min_distance = None

    def detect_contact(self):
        """Detects a contact with a rigid object."""
        body, link = self.body, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:  # pylint: disable=bare-except
                self.contact_constraint = None
                pass

        # Get all contact points between the suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        # print(points)
        # exit()
        if self.activated:
            points = [point for point in points if point[2] != self.body]

        # # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True

        return False

    def check_grasp(self):
        """Check a grasp (object in contact?) for picking success."""

        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object is not None


'''
Ver.2

class FingeredGripper(Gripper):
    """Simulate simple fingered gripper dynamics using Robotiq 2F-85."""

    def __init__(self, assets_root, robot, ee, obj_ids):
        """Initialize the gripper, loading the URDF and connecting it to the robot."""
        super().__init__(assets_root)
        self.robot_body_id = robot
        # Load the Robotiq gripper base model from URDF.
        gripper_urdf = os.path.join(self.assets_root, GRIPPER_URDF)
        self.body = pybullet_utils.load_urdf(p, gripper_urdf, basePosition=(0.487,0.109,0.438), baseOrientation=p.getQuaternionFromEuler((np.pi, 0, 0)))

        # Attach the gripper to the robot's end effector.
        p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=self.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))

        # Reference to object IDs in the environment for gripping simulation.
        self.obj_ids = obj_ids
        self.activated = False

        # Track constraints and anchors for gripping objects (rigid and deformable).
        self.contact_constraint = None
        self.def_grip_item = None
        self.def_grip_anchors = []
        
        # object that the robot has to grasp and pick/place
        self.target_object = None
        
        # # Set friction coefficients for gripper fingers
        # for i in range(p.getNumJoints(self.body)):
        #     p.changeDynamics(self.body, i, lateralFriction=1.0, spinningFriction=1.0,
        #                      rollingFriction=0.0001, frictionAnchor=True)

        # self.step_simulation(1e3)
        self.setup_mimic_joints()
        
        self.open_gripper()

    def step_simulation(self, num_steps):
        for _ in range(int(num_steps)):
            p.stepSimulation()
            if self.body is not None:
                gripper_joint_position = p.getJointState(self.body, 1)[0]
                p.setJointMotorControl2(self.body, 1, p.POSITION_CONTROL, targetPosition=gripper_joint_position)
            time.sleep(1e-3)

    def setup_mimic_joints(self):
        # constraints to make joints 6, 3, 8, 5, 10 mimic joint 1
        joint_indices = [6, 3, 8, 5, 10]
        for i, joint in enumerate(joint_indices):
            constraint_id = p.createConstraint(
                self.body, 1, self.body, joint, 
                jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], #jointAxis=[0, 1, 0]
                parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]
            )
            gear_ratio = -1 if joint in [3, 8] else 1
            p.changeConstraint(constraint_id, gearRatio=gear_ratio, maxForce=10000, erp=1) # erp=1
        p.setRealTimeSimulation(1)

 
    def close_gripper(self):
        # p.setJointMotorControl2(self.body, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=10000)
        p.setJointMotorControl2(self.body, 1, p.POSITION_CONTROL, targetPosition=1.0, force=10000)
        # self.step_simulation(4e2) # 4e2
        
    def open_gripper(self):
        print("opened")
        # p.setJointMotorControl2(self.body, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
        p.setJointMotorControl2(self.body, 1, p.POSITION_CONTROL, targetPosition=0.0)
        # self.step_simulation(4e2)

    def activate(self):
        """Activate the gripper to grasp an object."""
        if not self.activated:
            self.close_gripper()
            
            points = p.getContactPoints(bodyA=self.body, linkIndexA=4)
            print(len(points))
            if points:
                find_object = None
                for point in points:
                    if self.target_object == point[2]:
                        find_object = point
                
                if find_object is None:
                    find_object = points[0]
                    print("<---- can't grasp object ---->")

                obj_id, contact_link = find_object[2], find_object[4]
                body_pose = p.getLinkState(self.body, 0) # p.getLinkState(self.body, 0) -> 4 is correct but it connects with the ground..
                obj_pose = p.getBasePositionAndOrientation(obj_id)
                world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])

                self.contact_constraint = p.createConstraint(
                    parentBodyUniqueId=self.body,
                    parentLinkIndex=0,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=contact_link,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=obj_to_body[0],
                    parentFrameOrientation=obj_to_body[1],
                    childFramePosition=(0, 0, 0),
                    childFrameOrientation=(0, 0, 0))

                self.activated = True

    def release(self):
        """Release any gripped object."""
        if self.activated:
            self.activated = False

            # Release rigid object.
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:  # pylint: disable=bare-except
                    pass

            # Open the fingers by resetting the revolute joints
            self.open_gripper()

            # Reset variables for deformables and other constraints.
            self.def_grip_anchors = []
            self.def_grip_item = None

    def detect_contact(self):
        """Detect contact with a rigid object."""
        threshold = 0.05
        link = 0
        body, right_ink, left_link = self.body, 4, 9
        closest_distance, closest_object = float('inf'), None
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:  # pylint: disable=bare-except
                self.contact_constraint = None
                
        points = p.getContactPoints(bodyA=body, linkIndexA=link)

        for obj_id in self.obj_ids['rigid']:
            # closest points for fingers
            right_finger_points = p.getClosestPoints(bodyA=body, bodyB=obj_id, linkIndexA=right_ink, distance=threshold)
            left_finger_points = p.getClosestPoints(bodyA=body, bodyB=obj_id, linkIndexA=left_link, distance=threshold)

            for point in right_finger_points: # each element of list 
                distance = point[8] # return value (8th element = ContactDistance)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_object = obj_id

            for point in left_finger_points:
                distance = point[8]
                if distance < closest_distance:
                    closest_distance = distance
                    closest_object = obj_id

        if self.activated:
            points = [point for point in points if point[2] != self.body]
            self.target_object = points
        else:
            self.target_object = closest_object
        
        return self.target_object

    def check_grasp(self):
        """Check if the gripper has successfully grasped an object."""
        contact_object = None
        if self.contact_constraint is not None:
            contact_object = p.getConstraintInfo(self.contact_constraint)[2]
        return contact_object is not None
    
'''

# class FingeredGripper(Gripper):
#     """Simulate simple fingered gripper dynamics using Robotiq 2F-85."""

#     def __init__(self, assets_root, robot, ee, obj_ids):
#         """Initialize the gripper, loading the URDF and connecting it to the robot."""
#         super().__init__(assets_root)

#         # Load the Robotiq gripper base model from URDF.
#         gripper_urdf = os.path.join(self.assets_root, GRIPPER_URDF)
#         self.body = pybullet_utils.load_urdf(p,
#                                              gripper_urdf,
#                                              basePosition=(0.487,0.109,0.438),
#                                              baseOrientation=p.getQuaternionFromEuler((np.pi, 0, 0)))

#         # Attach the gripper to the robot's end effector.
#         p.createConstraint(
#             parentBodyUniqueId=robot,
#             parentLinkIndex=ee,
#             childBodyUniqueId=self.body,
#             childLinkIndex=-1,
#             jointType=p.JOINT_FIXED,
#             jointAxis=(0, 0, 0),
#             parentFramePosition=(0, 0, 0),
#             childFramePosition=(0, 0, 0.01))

#         # Reference to object IDs in the environment for gripping simulation.
#         self.obj_ids = obj_ids
#         self.activated = False

#         # Track constraints and anchors for gripping objects (rigid and deformable).
#         self.contact_constraint = None
#         self.def_grip_item = None
#         self.def_grip_anchors = []
        
#         # object that the robot has to grasp and pick/place
#         self.target_object = None
        
#         # Set friction coefficients for gripper fingers
#         # for i in range(p.getNumJoints(self.body)):
#         #     p.changeDynamics(self.body, i, lateralFriction=1.0, spinningFriction=1.0,
#         #                      rollingFriction=0.0001, frictionAnchor=True)
#         # self.step_simulation(1e3)
        
#         # self.setup_mimic_joints()
#         self.open_gripper()

#     def step_simulation(self, num_steps):
#         for i in range(int(num_steps)):
#             p.stepSimulation()
#             if self.body is not None:
#                 # Constraints
#                 gripper_joint_positions = np.array([p.getJointState(self.body, i)[0] for i in range(p.getNumJoints(self.body))])
#                 p.setJointMotorControlArray(
#                     self.body, [6, 3, 8, 5, 10], p.POSITION_CONTROL,
#                     [
#                         gripper_joint_positions[1], -gripper_joint_positions[1], 
#                         -gripper_joint_positions[1], gripper_joint_positions[1],
#                         gripper_joint_positions[1]
#                     ],
#                     positionGains=np.ones(5)
#                 )
#             # time.sleep(1e-2)

#     def setup_mimic_joints(self):
#         # constraints to make joints 6, 3, 8, 5, 10 mimic joint 1
#         joint_indices = [6, 3, 8, 5, 10]
#         for i, joint in enumerate(joint_indices):
#             constraint_id = p.createConstraint(
#                 self.body, 1, self.body, joint, 
#                 jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], #jointAxis=[0, 1, 0]
#                 parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]
#             )
#             gear_ratio = -1 if joint in [3, 8] else 1
#             p.changeConstraint(constraint_id, gearRatio=gear_ratio, maxForce=1000, erp=1) # erp=1
#         # p.setRealTimeSimulation(1)

#     def close_gripper(self):
#         # p.setJointMotorControl2(self.body, 1, p.VELOCITY_CONTROL, targetVelocity=5, force=50000)
#         p.setJointMotorControl2(self.body, 1, p.POSITION_CONTROL, targetPosition=1.0, force=50000)
#         # p.setJointMotorControl2(self.body, 6, p.POSITION_CONTROL, targetPosition=1.0)
#         self.step_simulation(4e2)
        
#     def open_gripper(self):
#         # p.setJointMotorControl2(self.body, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
#         p.setJointMotorControl2(self.body, 1, p.POSITION_CONTROL, targetPosition=0.0, force=50000)
#         self.step_simulation(4e2)

#     def activate(self):
#         """Activate the gripper to grasp an object."""
#         if not self.activated:
#             self.close_gripper()

#             points = p.getContactPoints(bodyA=self.body, linkIndexA=4)
#             print(len(points))
#             if points:
#                 find_object = None
#                 for point in points:
#                     if self.target_object == point[2]:
#                         find_object = point
                
#                 if find_object is None:
#                     find_object = points[0]
#                     print("<---- can't grasp object ---->")

#                 obj_id, contact_link = find_object[2], find_object[4]
#                 body_pose = p.getLinkState(self.body, 0) # p.getLinkState(self.body, 0) -> 4 is correct but it connects with the ground..
#                 obj_pose = p.getBasePositionAndOrientation(obj_id)
#                 world_to_body = p.invertTransform(body_pose[0], body_pose[1])
#                 obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])

#                 self.contact_constraint = p.createConstraint(
#                     parentBodyUniqueId=self.body,
#                     parentLinkIndex=0,
#                     childBodyUniqueId=obj_id,
#                     childLinkIndex=contact_link,
#                     jointType=p.JOINT_FIXED,
#                     jointAxis=(0, 0, 0),
#                     parentFramePosition=obj_to_body[0],
#                     parentFrameOrientation=obj_to_body[1],
#                     childFramePosition=(0, 0, 0),
#                     childFrameOrientation=(0, 0, 0))

#                 self.activated = True

#     def release(self):
#         """Release any gripped object."""
#         if self.activated:
#             self.activated = False

#             # Release rigid object.
#             if self.contact_constraint is not None:
#                 try:
#                     p.removeConstraint(self.contact_constraint)
#                     self.contact_constraint = None
#                 except:  # pylint: disable=bare-except
#                     pass

#             # Open the fingers by resetting the revolute joints
#             self.open_gripper()

#             # Reset variables for deformables and other constraints.
#             self.def_grip_anchors = []
#             self.def_grip_item = None

#     def detect_contact(self):
#         """Detect contact with a rigid object."""
#         threshold = 0.05
#         link = 0
#         body, right_ink, left_link = self.body, 4, 9
#         closest_distance, closest_object = float('inf'), None
#         if self.activated and self.contact_constraint is not None:
#             try:
#                 info = p.getConstraintInfo(self.contact_constraint)
#                 body, link = info[2], info[3]
#             except:  # pylint: disable=bare-except
#                 self.contact_constraint = None
                
#         points = p.getContactPoints(bodyA=body, linkIndexA=link)

#         for obj_id in self.obj_ids['rigid']:
#             # closest points for fingers
#             right_finger_points = p.getClosestPoints(bodyA=body, bodyB=obj_id, linkIndexA=right_ink, distance=threshold)
#             left_finger_points = p.getClosestPoints(bodyA=body, bodyB=obj_id, linkIndexA=left_link, distance=threshold)

#             for point in right_finger_points: # each element of list 
#                 distance = point[8] # return value (8th element = ContactDistance)
#                 if distance < closest_distance:
#                     closest_distance = distance
#                     closest_object = obj_id

#             for point in left_finger_points:
#                 distance = point[8]
#                 if distance < closest_distance:
#                     closest_distance = distance
#                     closest_object = obj_id

#         if self.activated:
#             points = [point for point in points if point[2] != self.body]
#             self.target_object = points
#         else:
#             self.target_object = closest_object
        
#         return self.target_object

#     def check_grasp(self):
#         """Check if the gripper has successfully grasped an object."""
#         contact_object = None
#         if self.contact_constraint is not None:
#             contact_object = p.getConstraintInfo(self.contact_constraint)[2]
#         return contact_object is not None

"""Classes to handle gripper dynamics."""

import os

import numpy as np
from cliport.utils import pybullet_utils

import pybullet as p

SPATULA_BASE_URDF = 'ur5/spatula/spatula-base.urdf'
SUCTION_BASE_URDF = 'ur5/suction/suction-base.urdf'
SUCTION_HEAD_URDF = 'ur5/suction/suction-head.urdf'
GRIPPER_URDF = 'ur5/gripper/robotiq_2f_85.urdf'
UR5_URDF_PATH = 'ur5/ur5.urdf'

class Gripper:
    """Base gripper class."""

    def __init__(self, assets_root):
        self.assets_root = assets_root
        self.activated = False

    def step(self):
        """This function can be used to create gripper-specific behaviors."""
        return

    def activate(self, objects):
        del objects
        return

    def release(self):
        return


class Spatula(Gripper):
    """Simulate simple spatula for pushing."""

    def __init__(self, assets_root, robot, ee, obj_ids):  # pylint: disable=unused-argument
        """Creates spatula and 'attaches' it to the robot."""
        super().__init__(assets_root)

        # Load spatula model.
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, SPATULA_BASE_URDF), pose[0], pose[1])
        p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))


class Suction(Gripper):
    """Simulate simple suction dynamics."""

    def __init__(self, assets_root, robot, ee, obj_ids):
        """Creates suction and 'attaches' it to the robot.
    
        Has special cases when dealing with rigid vs deformables. For rigid,
        only need to check contact_constraint for any constraint. For soft
        bodies (i.e., cloth or bags), use cloth_threshold to check distances
        from gripper body (self.body) to any vertex in the cloth mesh. We
        need correct code logic to handle gripping potentially a rigid or a
        deformable (and similarly for releasing).
    
        To be clear on terminology: 'deformable' here should be interpreted
        as a PyBullet 'softBody', which includes cloths and bags. There's
        also cables, but those are formed by connecting rigid body beads, so
        they can use standard 'rigid body' grasping code.
    
        To get the suction gripper pose, use p.getLinkState(self.body, 0),
        and not p.getBasePositionAndOrientation(self.body) as the latter is
        about z=0.03m higher and empirically seems worse.
    
        Args:
          assets_root: str for root directory with assets.
          robot: int representing PyBullet ID of robot.
          ee: int representing PyBullet ID of end effector link.
          obj_ids: list of PyBullet IDs of all suctionable objects in the env.
        """
        super().__init__(assets_root)

        # Load gripper base model (visual only).
        pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
        base = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
        p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=base,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))

        # Load suction tip model (visual and collision) with compliance.
        # urdf = 'assets/ur5/suction/suction-head.urdf'
        pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
        self.body = pybullet_utils.load_urdf(
            p, os.path.join(self.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])
        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=self.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.08))
        p.changeConstraint(constraint_id, maxForce=50)

        # Reference to object IDs in environment for simulating suction.
        self.obj_ids = obj_ids

        # Indicates whether gripper is gripping anything (rigid or def).
        self.activated = False

        # For gripping and releasing rigid objects.
        self.contact_constraint = None

        # Defaults for deformable parameters, and can override in tasks.
        self.def_ignore = 0.035  # TODO(daniel) check if this is needed
        self.def_threshold = 0.030
        self.def_nb_anchors = 1

        # Track which deformable is being gripped (if any), and anchors.
        self.def_grip_item = None
        self.def_grip_anchors = []

        # Determines release when gripped deformable touches a rigid/def.
        # TODO(daniel) should check if the code uses this -- not sure?
        self.def_min_vetex = None
        self.def_min_distance = None

        # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
        self.init_grip_distance = None
        self.init_grip_item = None

    def activate(self):
        """Simulate suction using a rigid fixed constraint to contacted object."""
        # TODO(andyzeng): check deformables logic.
        # del def_ids

        if not self.activated:
            points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
            # print(points)
            if points:

                # Handle contact between suction with a rigid object.
                for point in points:
                    obj_id, contact_link = point[2], point[4]
                if obj_id in self.obj_ids['rigid']:
                    body_pose = p.getLinkState(self.body, 0)
                    obj_pose = p.getBasePositionAndOrientation(obj_id)
                    world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                       world_to_body[1],
                                                       obj_pose[0], obj_pose[1])
                    self.contact_constraint = p.createConstraint(
                        parentBodyUniqueId=self.body,
                        parentLinkIndex=0,
                        childBodyUniqueId=obj_id,
                        childLinkIndex=contact_link,
                        jointType=p.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0))

                self.activated = True

    def release(self):
        """Release gripper object, only applied if gripper is 'activated'.
    
        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.
    
        To handle deformables, simply remove constraints (i.e., anchors).
        Also reset any relevant variables, e.g., if releasing a rigid, we
        should reset init_grip values back to None, which will be re-assigned
        in any subsequent grasps.
        """
        if self.activated:
            self.activated = False

            # Release gripped rigid object (if any).
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:  # pylint: disable=bare-except
                    pass
                self.init_grip_distance = None
                self.init_grip_item = None

            # Release gripped deformable object (if any).
            if self.def_grip_anchors:
                for anchor_id in self.def_grip_anchors:
                    p.removeConstraint(anchor_id)
                self.def_grip_anchors = []
                self.def_grip_item = None
                self.def_min_vetex = None
                self.def_min_distance = None

    def detect_contact(self):
        """Detects a contact with a rigid object."""
        body, link = self.body, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:  # pylint: disable=bare-except
                self.contact_constraint = None
                pass

        # Get all contact points between the suction and a rigid body.
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        # print(points)
        # exit()
        if self.activated:
            points = [point for point in points if point[2] != self.body]

        # # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True

        return False

    def check_grasp(self):
        """Check a grasp (object in contact?) for picking success."""

        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object is not None
    
class FingeredGripper(Gripper):
    """Simulate simple fingered gripper dynamics using Robotiq 2F-85."""

    def __init__(self, assets_root, robot, ee, obj_ids):
        """Initialize the gripper, loading the URDF and connecting it to the robot."""
        super().__init__(assets_root)

        # Load the Robotiq gripper base model from URDF.
        gripper_urdf = os.path.join(self.assets_root, GRIPPER_URDF)
        self.body = pybullet_utils.load_urdf(p, gripper_urdf, basePosition=(0.487,0.109,0.438), baseOrientation=p.getQuaternionFromEuler((np.pi, 0, 0)))

        # Attach the gripper to the robot's end effector.
        p.createConstraint(
            parentBodyUniqueId=robot,
            parentLinkIndex=ee,
            childBodyUniqueId=self.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0.01))

        # Reference to object IDs in the environment for gripping simulation.
        self.obj_ids = obj_ids
        self.activated = False

        # Track constraints and anchors for gripping objects (rigid and deformable).
        self.contact_constraint = None
        self.def_grip_item = None
        self.def_grip_anchors = []
        
        # object that the robot has to grasp and pick/place
        self.target_object = None
        
        # # Set friction coefficients for gripper fingers
        # for i in range(p.getNumJoints(self.body)):
        #     p.changeDynamics(self.body, i, lateralFriction=1.0, spinningFriction=1.0,
        #                      rollingFriction=0.0001, frictionAnchor=True)
        # self.step_simulation(1e3)
        
        self.open_gripper()

    def step_simulation(self, num_steps):
        for i in range(int(num_steps)):
            p.stepSimulation()
            if self.body is not None:
                # Constraints
                gripper_joint_positions = np.array([p.getJointState(self.body, i)[0] for i in range(p.getNumJoints(self.body))])
                p.setJointMotorControlArray(
                    self.body, [6, 3, 8, 5, 10], p.POSITION_CONTROL,
                    [
                        gripper_joint_positions[1], -gripper_joint_positions[1], 
                        -gripper_joint_positions[1], gripper_joint_positions[1],
                        gripper_joint_positions[1]
                    ],
                    positionGains=np.ones(5)
                )

    def close_gripper(self):
        # p.setJointMotorControl2(self.body, 1, p.VELOCITY_CONTROL, targetVelocity=10, force=10000)
        p.setJointMotorControl2(self.body, 1, p.POSITION_CONTROL, targetPosition=1.0, force=10000)
        # p.setJointMotorControl2(self.body, 6, p.POSITION_CONTROL, targetPosition=1.0)

        self.step_simulation(4e2)
        
    def open_gripper(self):
        # p.setJointMotorControl2(self.body, 1, p.VELOCITY_CONTROL, targetVelocity=-5, force=10000)
        p.setJointMotorControl2(self.body, 1, p.POSITION_CONTROL, targetPosition=0.0)
        self.step_simulation(4e2)

    def activate(self):
        """Activate the gripper to grasp an object."""
        if not self.activated:
            self.close_gripper()
            
            points = p.getContactPoints(bodyA=self.body, linkIndexA=4)
            print(len(points))
            if points:
                find_object = None
                for point in points:
                    if self.target_object == point[2]:
                        find_object = point
                
                if find_object is None:
                    find_object = points[0]
                    print("<---- can't grasp object ---->")

                obj_id, contact_link = find_object[2], find_object[4]
                body_pose = p.getLinkState(self.body, 0) # p.getLinkState(self.body, 0) -> 4 is correct but it connects with the ground..
                obj_pose = p.getBasePositionAndOrientation(obj_id)
                world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                obj_to_body = p.multiplyTransforms(world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])

                self.contact_constraint = p.createConstraint(
                    parentBodyUniqueId=self.body,
                    parentLinkIndex=0,
                    childBodyUniqueId=obj_id,
                    childLinkIndex=contact_link,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=obj_to_body[0],
                    parentFrameOrientation=obj_to_body[1],
                    childFramePosition=(0, 0, 0),
                    childFrameOrientation=(0, 0, 0))

                self.activated = True

    def release(self):
        """Release any gripped object."""
        if self.activated:
            self.activated = False

            # Release rigid object.
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:  # pylint: disable=bare-except
                    pass

            # Open the fingers by resetting the revolute joints
            self.open_gripper()

            # Reset variables for deformables and other constraints.
            self.def_grip_anchors = []
            self.def_grip_item = None

    def detect_contact(self):
        """Detect contact with a rigid object."""
        threshold = 0.05
        link = 0
        body, right_ink, left_link = self.body, 4, 9
        closest_distance, closest_object = float('inf'), None
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:  # pylint: disable=bare-except
                self.contact_constraint = None
                
        points = p.getContactPoints(bodyA=body, linkIndexA=link)

        for obj_id in self.obj_ids['rigid']:
            # closest points for fingers
            right_finger_points = p.getClosestPoints(bodyA=body, bodyB=obj_id, linkIndexA=right_ink, distance=threshold)
            left_finger_points = p.getClosestPoints(bodyA=body, bodyB=obj_id, linkIndexA=left_link, distance=threshold)

            for point in right_finger_points: # each element of list 
                distance = point[8] # return value (8th element = ContactDistance)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_object = obj_id

            for point in left_finger_points:
                distance = point[8]
                if distance < closest_distance:
                    closest_distance = distance
                    closest_object = obj_id

        if self.activated:
            points = [point for point in points if point[2] != self.body]
            self.target_object = points
        else:
            self.target_object = closest_object
        
        return self.target_object

    def check_grasp(self):
        """Check if the gripper has successfully grasped an object."""
        contact_object = None
        if self.contact_constraint is not None:
            contact_object = p.getConstraintInfo(self.contact_constraint)[2]
        return contact_object is not None