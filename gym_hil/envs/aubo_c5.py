import os
file_path = os.path.dirname(os.path.abspath(__file__))

import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
@register_agent()
class AuboC5(BaseAgent):
    uid = "aubo_c5"
    urdf_path = os.path.join(file_path, "../assets/auboc5/urdf/aubo_C5.urdf")
    keyframes = dict(
        reset=Keyframe(
            qpos=np.array(
                [1.5623385,  0.7971761,  1.6842455,  0.8892992,  0.01736733, 0.01151336]
            ),
            pose=sapien.Pose(),
        )
    )
    arm_joint_names = [
        "shoulder_joint",
        "upperArm_joint",
        "foreArm_joint",
        "wrist1_joint",
        "wrist2_joint",
        "wrist3_joint"
    ]

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 1000

    @property
    def _controller_configs(self):
        arm_pdee_delta_pos = PDEEPoseControllerConfig(
            self.arm_joint_names,
            ee_link = "charging_gun_Link",
            pos_lower=[-1.0] * 3,
            pos_upper=[1.0] * 3,
            rot_lower=-1.0,
            rot_upper=1.0,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=False
        )


        controller_configs = dict(
            pdee_delta_pos=dict(
                arm=arm_pdee_delta_pos
            )
            
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)