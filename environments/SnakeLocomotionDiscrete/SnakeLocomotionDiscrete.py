import numpy as np
import os

import pybullet as p

from somo.sm_manipulator_definition import SMManipulatorDefinition
from somo.sm_continuum_manipulator import SMContinuumManipulator

from environments import SomoEnv
# インポート文の修正
#from somogym.environments.SomoEnv import SomoEnv


###################################################
import shutil
import time
from multiprocessing import Lock

# ロックの作成（オプション）
delete_lock = Lock()

def safe_remove(urdf_path):
    """URDFファイルを安全に削除する関数。最大5回試行。環境の並列化による競合を防止"""
    for i in range(5):  # 最大5回試みる
        try:
            os.remove(urdf_path)
            print(f"Successfully deleted {urdf_path}")
            break
        except FileNotFoundError:
            print(f"FileNotFoundError: {urdf_path} は既に存在しません。")
            break
        except PermissionError:
            print(f"PermissionError: {urdf_path} を削除できません。リトライします... (Attempt {i+1}/5)")
            time.sleep(1)  # 1秒待機してリトライ
        except Exception as e:
            print(f"Error deleting {urdf_path}: {e}")
            break
###################################################



#class SnakeLocomotionDiscrete(SomoEnv):
class SnakeLocomotionDiscrete(SomoEnv.SomoEnv):
    def __init__(self, run_config=None, run_ID=None, render=False, debug=0):

        # load the manipulator definition
        manipulator_def_file = os.path.join(
            os.path.dirname(__file__), "definitions/snake_discrete.yaml"
        )
        self.manipulator_def = SMManipulatorDefinition.from_file(manipulator_def_file)

        
        # ユニークなURDFファイル名を生成（run_IDを使用）
        unique_urdf_filename = f"{self.manipulator_def.manipulator_name}_{'_'.join(run_ID)}.urdf"
        self.manipulator_def.urdf_filename = os.path.join(
            os.path.dirname(__file__),
            "definitions",
            unique_urdf_filename,
        )

        # デバッグ用プリント
        print(f"URDF filename set to {self.manipulator_def.urdf_filename}")
        print(f"Definitions directory: {os.path.dirname(self.manipulator_def.urdf_filename)}")


        #manipulator_def.urdf_filename = os.path.join(
        #    os.path.dirname(__file__),
        #    "definitions/" + manipulator_def.manipulator_name + ".urdf",
        #)

        active_actuators_list = [
            (
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
                [
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ],
            )
        ]
        manipulators = [SMContinuumManipulator(self.manipulator_def)]

        start_positions = [[-8, 0, 0.5]]
        start_orientations = [p.getQuaternionFromEuler([0, np.pi / 2, 0])]
        manipulator_base_constraints = ["free"]

        action_shape = (16,)

        if run_config["difficulty"] == "basic":
            self.randomize_target = False
            self.target_pos = np.array([60, 0, 0])
            self.target_resample_probability = 0
        elif run_config["difficulty"] == "advanced":
            self.randomize_target = True
            self.target_pos = np.array([60, 15, 0])
            self.target_resample_probability = run_config["target_resample_probability"]

        self.target_range = {"min": [-70, -70, 0], "max": [70, 70, 0]}
        self.target_id = None # 追加

        # 各報酬コンポーネントの情報を記録する辞書を初期化
        self.reward_component_info = {
            "U": 0.0,
            "v_g": 0.0,
            "goal_range_reward": 0.0,
            #"head_target_dist": 0.0  # 報酬関数に応じて使い分ける
        }
        
        super().__init__(
            manipulators=manipulators,
            start_positions=start_positions,
            start_orientations=start_orientations,
            manipulator_base_constraints=manipulator_base_constraints,
            active_actuators_list=active_actuators_list,
            action_shape=action_shape,
            run_config=run_config,
            gravity=[0, 0, -196.2],  # scale factor 20 applied to gravity
            ground_plane_height=0,
            run_ID=run_ID,
            render=render,
            debug=debug,
        )

    def step(self, action):

        # check whether we need to resample the target position # todo: why is this task being done here? is this necessary at all? if so, should it be a separate fn?
        if (
            self.randomize_target
            and not self.stabilizing
            and self.np_random.uniform(0, 1) < self.target_resample_probability
        ):
            # resampling target position
            self.target_pos = self.np_random.uniform(
                self.target_range["min"], self.target_range["max"]
            )
            # relaoding target position
            object_urdf_path = os.path.join(
                os.path.dirname(__file__),
                "definitions/additional_urdfs/target_sphere.urdf",
            )
            # todo: replace try / except with proper handling of the different cases for self.target_id
            try:
                p.removeBody(
                    self.target_id,
                    physicsClientId=self.physics_client,
                )
            except:
                pass
            self.target_id = p.loadURDF(
                object_urdf_path,
                self.target_pos,
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=1,
                physicsClientId=self.physics_client,
            )

        obs, reward, done, info = super().step(action=action)
        return (self.get_observation(), reward, done, info)

        # todo: check if ground plane has friction; it does

    def get_observation_shape(self):
        obs_dimension_per_sample = {
            "target_pos": 3,
            "error": 3,
            "error_distance": 1,  # distance to the goal
            "error_direction": 3,  # distance to the goal
            "positions": 3,
            "velocities": 3,
            "tip_pos": 3,
            "angles": 1,
            "curvatures": 1,
            "applied_input_torques": 16,
        }
        obs_flags = self.run_config["observation_flags"]
        obs_len = 0
        for f in obs_flags:
            num_pts = obs_flags[f] if obs_flags[f] is not None else 1
            obs_len += num_pts * obs_dimension_per_sample[f]
        return (obs_len,)

    def get_observation(self):
        obs_flags = self.run_config["observation_flags"]
        self.manipulator_states = self.get_manipulator_states(
            components=list(obs_flags.keys())
        )
        state = np.array([])

        if "target_pos" in obs_flags:
            state = np.concatenate((state, self.target_pos))

        if "positions" in obs_flags:
            positions = np.array(
                [state["positions"] for state in self.manipulator_states]
            )
            if obs_flags["positions"]:
                positions = np.array(
                    [
                        self.reduce_state_len(ps, obs_flags["positions"])
                        for ps in positions
                    ]
                )
            state = np.concatenate((state, positions.flatten()))

        if "velocities" in obs_flags:
            velocities = np.array(
                [state["velocities"] for state in self.manipulator_states]
            )
            if obs_flags["velocities"]:
                velocities = np.array(
                    [
                        self.reduce_state_len(vs, obs_flags["velocities"])
                        for vs in velocities
                    ]
                )
            state = np.concatenate((state, velocities.flatten()))

        if (
            "tip_pos" in obs_flags
            or "error" in obs_flags
            or "error_distance" in obs_flags
            or "error_direction" in obs_flags
        ):
            positions = np.array(
                [state["positions"] for state in self.manipulator_states]
            )
            tip_pos = positions[:, -1].flatten()
            if "tip_pos" in obs_flags:
                state = np.concatenate((state, tip_pos))
            if "error" in obs_flags:
                state = np.concatenate((state, self.target_pos - tip_pos))
            if "error_distance" in obs_flags:
                state = np.concatenate(
                    (state, np.array([np.linalg.norm(self.target_pos - tip_pos)]))
                )
            if "error_direction" in obs_flags:
                state = np.concatenate(
                    (
                        state,
                        (self.target_pos - tip_pos)
                        / np.linalg.norm(self.target_pos - tip_pos),
                    )
                )

        if "angles" in obs_flags:
            # todo: this backbone angle stuff seems to have a problem... rethink how to read this - maybe get joint angles instead. try to get to angles that are 0 in one of the axes in case of planar manipulators
            angles = np.array([state["angles"] for state in self.manipulator_states])
            if obs_flags["angles"]:
                angles = np.array(
                    [
                        self.reduce_state_len(angs, obs_flags["angles"])
                        for angs in angles
                    ]
                )
            state = np.concatenate((state, angles.flatten()))

        if "curvatures" in obs_flags:
            curvatures = np.array(
                [state["curvatures"] for state in self.manipulator_states]
            )
            if obs_flags["curvatures"]:
                curvatures = np.array(
                    [
                        self.reduce_state_len(cs, obs_flags["curvatures"])
                        for cs in curvatures
                    ]
                )
            state = np.concatenate((state, curvatures.flatten()))

        if "applied_input_torques" in obs_flags:
            applied_input_torques = np.array(self.applied_torque)
            state = np.concatenate((state, applied_input_torques.flatten()))

        return state

        # if self.manipulators[0].instantiated:

        #     backbone_pos = np.array(self.manipulators[0].get_backbone_positions())
        #     backbone_pos_reshaped = np.reshape(backbone_pos, (-1))
        #     backbone_curves = np.array(self.manipulators[0].get_backbone_curvatures())
        #     head_pos = backbone_pos[-1, :]
        #     head_pos_reshaped = np.reshape(head_pos, (-1))
        #     target_pos = self.target_pos
        #     error = target_pos - head_pos

        #     state = np.concatenate(
        #         (
        #             backbone_pos_reshaped,
        #             backbone_curves,
        #             self.applied_torque,
        #             head_pos_reshaped,
        #             target_pos,
        #             error,
        #         )
        #     )

        #     return state

        # else:
        #     obs_len = 153
        #     return np.array([None] * obs_len)

    # def get_reward(self, *args, **kwargs):    #　こっちの報酬関数を使う場合はself.reward_component_infoのキーを変更する
    #     if self.stabilizing:
    #         return 0

    #     reward_flags = self.run_config["reward_flags"]
    #     head_pos = self.manipulator_states[0]["positions"][-1]
    #     target_pos = self.target_pos
    #     reward = 0

    #     if "head_target_dist" in reward_flags:
    #         head_target_dist_reward = np.linalg.norm(target_pos - head_pos)
    #         self.reward_component_info["head_target_dist"] += head_target_dist_reward
    #         reward += reward_flags["head_target_dist"] * head_target_dist_reward

    #     return reward
    

    def get_reward(self, *args, **kwargs):  # ポテンシャル場を用いた報酬関数
        if self.stabilizing:
            return 0

        """
        # 定数の設定
        c_v : cvの定数（速度の射影に対する重み）
        c_g : cgの定数（ゴール到達報酬の重み）
        r_k_values : basicタスクに基づくゴール範囲
        """

        # run_configから報酬に必要なフラグと定数を取得
        reward_flags = self.run_config.get("reward_flags", {})
        if not reward_flags:
            raise ValueError("`reward_flags` が設定ファイルに存在しません。")

        c_v = reward_flags.get("c_v", 1.0)            # 速度の射影に対する重み
        c_g = reward_flags.get("c_g", 1.0)            # ゴール到達報酬の重み
        r_k_values = reward_flags.get("r_k_values", [])

        if not r_k_values:
            raise ValueError("`r_k_values` が設定ファイルに存在しません。少なくとも1つのしきい値が必要です。")

        # manipulator_statesから頭の位置と速度を取得
        head_state = self.manipulator_states[0]
        head_pos = head_state["positions"][-1]  # 蛇の頭の位置
        v_s = head_state["velocities"][-1]      # 頭の速度ベクトル

        target_pos = self.target_pos  # ゴールの位置
        e_g = target_pos - head_pos   # ゴールと頭の位置の差ベクトル
        rho_g = np.linalg.norm(e_g)   # 頭からゴールまでの距離

        # ゼロ除算を防ぐためのチェック
        if rho_g == 0:
            f_g = np.zeros_like(e_g)
        else:
            f_g = e_g / rho_g  # ゴール方向の単位ベクトル

        # ポテンシャル場報酬 U の計算
        U = np.dot(v_s, f_g) / (rho_g if rho_g != 0 else 1.0)

        # ゴール方向の速度の射影 v_g の計算
        v_g = np.dot(v_s, f_g)

        # 角度 theta_g の計算
        norm_v_s = np.linalg.norm(v_s)
        if norm_v_s == 0 or rho_g == 0:
            theta_g = 0.0
        else:
            cos_theta = np.clip(np.dot(v_s, e_g) / (norm_v_s * rho_g), -1.0, 1.0)
            theta_g = np.arccos(cos_theta)

        # ゴール範囲報酬の計算
        indicator_sum = sum(1.0 / r_k for r_k in r_k_values if rho_g < r_k) # ゴールに近づくほど報酬が増加

        # 各報酬コンポーネントを記録（オプション）
        if hasattr(self, 'reward_component_info'):
            self.reward_component_info["U"] += U
            self.reward_component_info["v_g"] += v_g
            self.reward_component_info["goal_range_reward"] += indicator_sum

        # 最終的な報酬 R の計算
        reward = c_v * v_g + c_g * U + c_g * np.cos(theta_g) * indicator_sum

        return reward



    def modify_physics(self): 
        for manipulator in self.manipulators:
            anisotropicFriction = [0.01, 2, 0.01] # y方向の異方性摩擦力を大きく設定。実際の蛇の推進に似せるために横への移動（横滑り）をし辛くしていると考えられる。
            contact_properties = {
                "lateralFriction": 2.0, # 地面との摩擦係数
                # "restitution": 1.0,
                "anisotropicFriction": anisotropicFriction, # 異方性摩擦、特定の方向に対して摩擦力を変えることができる
            }
            manipulator.set_contact_property(contact_properties)

        object_urdf_path = os.path.join(
            os.path.dirname(__file__), "definitions/additional_urdfs/target_sphere.urdf"
        )
        # todo: improve this lazy try and except; manually handle all cases(実行時に存在しないボディを再度削除しようとしてエラーが出るので修正した。ただし、このエラー自体は無視しても問題ない)
        #try:
        #    p.removeBody(self.target_id)
        #except:
        #    pass
        # target_idが初期化されていない場合はスキップ
        if self.target_id is not None and p.getBodyUniqueId(self.target_id) != -1:
            p.removeBody(self.target_id, physicsClientId=self.physics_client)

        self.target_id = p.loadURDF(
            object_urdf_path,
            self.target_pos,
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=1,
            physicsClientId=self.physics_client,
        )

    def get_cam_settings(self):
        opt_str = "--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0"
        # cam_width, cam_height = 1920, 1640
        cam_width, cam_height = (
            None,
            None,
        )
        if cam_width is not None and cam_height is not None:
            opt_str += " --width=%d --height=%d" % (cam_width, cam_height)

        cam_distance, cam_yaw, cam_pitch, cam_xyz_target = (
            30.0,
            -15.0,
            -88.0,
            [25.0, 0.0, 0],
        )

        return opt_str, cam_distance, cam_yaw, cam_pitch, cam_xyz_target

    def check_success(self):
        head_pos = self.manipulator_states[0]["positions"][-1]
        target_pos = self.target_pos
        head_target_dist = np.linalg.norm(target_pos - head_pos)

        return head_target_dist <= 10
    

    def close(self):
        if getattr(self, 'physics_client', None) is not None:
            try:
                # シミュレーションステップを完了させる
                for _ in range(100):
                    p.stepSimulation(physicsClientId=self.physics_client)
                p.disconnect(physicsClientId=self.physics_client)
                print(f"Disconnected physics client: {self.physics_client}")
            except Exception as e:
                print(f"Error disconnecting physics client: {e}")
        else:
            print("Warning: physics_client is None, cannot disconnect.")

        self.first_run_done = False
        self.physics_client = None

        # URDF ファイルの削除は不要（definitionsディレクトリ内にある場合）
        # ただし、必要に応じて以下のコードを有効化
        # for manipulator in self.manipulators:
        #     urdf_path = manipulator.manipulator_definition.urdf_filename
        #     if os.path.isfile(urdf_path):
        #         # ロックを使用して削除処理を同期（オプション）
        #         with delete_lock:
        #             safe_remove(urdf_path)

        # definitionsディレクトリ内のURDFファイルを削除（必要な場合）
        # 例えば、run_IDを使ってユニークにした場合、以下のように削除する
        urdf_path = self.manipulator_def.urdf_filename
        if os.path.isfile(urdf_path):
            with delete_lock:
                safe_remove(urdf_path)

        print(f"Cleaned up URDF file at {urdf_path}")
