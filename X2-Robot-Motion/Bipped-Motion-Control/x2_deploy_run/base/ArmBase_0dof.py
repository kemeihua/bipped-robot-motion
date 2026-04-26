import os
import sys
from base.Base import *
from grpc import insecure_channel
from scripts.Config import Config
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../protos'))
from protos import arm_service_pb2_grpc as arm_pb2_grpc
from protos import droid_msg_pb2 as msg_pb2
from base.read_csv import load_action_sequences
import csv

class ArmBase:
    def __init__(self, _cfg):
        self.cfg = _cfg
        self.armActions = self.cfg.env.num_arm_actions
        # grpc defines
        self.armConfigs = msg_pb2.DroidConfigs()
        self.armState = msg_pb2.DroidArmResponse()
        self.armCommand = msg_pb2.DroidCommandRequest()
        init_command(self.armCommand, self.cfg.env.num_arm_actions)
        for idx in range(12):
            self.armCommand.finger.append(10)
        self.armSt = "init"  # 给定状态
        self.armSc = "init"  # 当前状态
        self.armStatus = "done"
        self.do_again = False
        self.dancing = "none"

    def get_arm_config(self):
        pass

    def get_arm_state(self):
        pass

    def set_arm_command(self):
        pass

    def arm_task(self, state):
        pass

    def arm_task_with_dance(self, state):
        pass

    def arm_task_str(self, arm_mode):
        pass

    def arm_task_str_with_dance(self, arm_mode):
        pass

    def do_arm_task(self):
        pass

    def walk_arm_swing(self, legState):
        pass

    def walk_arm_swing_6dof(self, legState):
        pass

if __name__ == '__main__':
    gBot = ArmBase(Config)


