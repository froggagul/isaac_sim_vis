from isaacsim import SimulationApp
import scipy.spatial
simulation_app = SimulationApp({"headless": False})

import os
import time
import numpy as np
import scipy
import pybullet_data
import random
import math
import glob
import copy
import pickle
from omni.syntheticdata import sensors
from isaacsim.sensors.camera import Camera
from omni.syntheticdata._syntheticdata import acquire_syntheticdata_interface, SensorType
# from omni.isaac.synthetic_utils import SyntheticDataHelper

from pxr import UsdShade, Sdf, Gf

from omni.kit.viewport.utility import frame_viewport_prims
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units, get_current_stage
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.api.robots import Robot
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.utils.viewports import set_camera_view, set_intrinsics_matrix
from isaacsim.core.prims import GeometryPrim

from pxr import Sdf, UsdLux, Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema

from omni.isaac.core.utils import prims
import omni.isaac.core.utils.prims as prim_utils
from omni.kit.viewport.utility import get_active_viewport, get_active_viewport_and_window, capture_viewport_to_file, capture_viewport_to_buffer
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, Writer
import omni.usd
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry

import carb.settings

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from PIL import Image

import broad_phase

import pybullet as p
import pybullet_data

class PBSCene:
    def __init__(self, config):

        # Extract camera parameters from USD camera prim
        viewport_api = get_active_viewport()
        camera_path = viewport_api.camera_path.pathString
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        camera = UsdGeom.Camera(camera_prim)
        gf_camera = camera.GetCamera()

        # Resolution
        width, height = viewport_api.resolution

        # Get camera transform (camera to world)
        cam_to_world = np.array(gf_camera.transform).reshape(4, 4).astype(np.float32)
        # Get intrinsic parameters
        focal_length = gf_camera.focalLength
        horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
        vertical_aperture = camera.GetVerticalApertureAttr().Get()
        near, far = camera.GetClippingRangeAttr().Get()

        # Store camera parameters for PyBullet
        self.cam_params = {
            'width': width,
            'height': height,
            'cam_to_world': cam_to_world,
            'focal_length': focal_length,
            'h_ap': horizontal_aperture,
            'v_ap': vertical_aperture,
            'near': near,
            'far': far
        }
        
        self.base_dir = '/home/dongwon/research/PointObjRep'

        self.build_scene(config)

    
    def build_scene(self, config):
        if p.isConnected():
            p.resetSimulation()  # if already connected, reset simulation
        else:
            p.connect(p.GUI)
        # Connect to PyBullet and set up environment

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        plane_uid = p.loadURDF("plane.urdf")
        p.resetBasePositionAndOrientation(plane_uid, [0,0,-0.2], [0,0,0,1.])

        # Load robot URDF from config
        robot_urdf = config["robot"]["urdf_path"]
        self.robot_uid = p.loadURDF(os.path.join(self.base_dir,robot_urdf), useFixedBase=True)
        base_pqc = config["robot"]['robot_base_pose']
        quaternion_scalar_first = base_pqc[3:]
        base_quat = np.concatenate([quaternion_scalar_first[1:], quaternion_scalar_first[:1]])
        p.resetBasePositionAndOrientation(self.robot_uid, base_pqc[:3], base_quat)
        self.robot_height = base_pqc[2]

        # load fixed objects
        if "mesh" in config:
            self.fixed_obj_uids = []
            for obj_key in config['mesh']:
                obj = config['mesh'][obj_key]
                mesh_path = os.path.join(self.base_dir, obj["file_path"])
                scale = obj.get("scale", 1.0)
                collision_id = p.createCollisionShape(
                    p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3
                )
                visual_id = p.createVisualShape(
                    p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3
                )
                quat_scalar_first = obj['pose'][3:]
                quat = np.concatenate([quat_scalar_first[1:], quat_scalar_first[:1]])
                body_uid = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision_id,
                    baseVisualShapeIndex=visual_id,
                    basePosition=obj['pose'][:3],
                    baseOrientation=quat,
                )
                self.fixed_obj_uids.append(body_uid)

        # Prepare lists for moving objects
        self.moving_obj_uids = []
        self.ee_link_indices = []
        self.ee_to_obj_pqc = []

        # Load each grasped object as a mesh body
        for obj in config["robot"].get("moving_obj", []):
            mesh_path = os.path.join(self.base_dir, obj["file_path"])
            scale = obj.get("scale", 1.0)
            collision_id = p.createCollisionShape(
                p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3
            )
            visual_id = p.createVisualShape(
                p.GEOM_MESH, fileName=mesh_path, meshScale=[scale]*3
            )
            body_uid = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
            )
            self.moving_obj_uids.append(body_uid)

            # Record end-effector link and relative transform
            self.ee_link_indices.append(obj["attached_to"])
            pq = obj["ee_to_obj_pq"]  # [x,y,z, qw,qx,qy,qz]
            ee_pos = pq[0:3]
            # convert scalar-first quat to PyBullet format [x,y,z,w]
            ee_ori = [pq[4], pq[5], pq[6], pq[3]]
            self.ee_to_obj_pqc.append((ee_pos, ee_ori))


    def get_imgs(self):
        # Compute view and projection matrices for PyBullet
        params = self.cam_params
        w, h = params['width'], params['height']
        cam2world = params['cam_to_world']
        # view_mat = np.linalg.inv(cam2world)

        # add frame debug visualizer at cam2world
        cam2world = cam2world.T
        origin = cam2world[:3, 3]
        x_axis = cam2world[:3, 0]
        y_axis = cam2world[:3, 1]
        z_axis = cam2world[:3, 2]
        axis_length = 0.1  # adjust length as needed

        p.addUserDebugLine(origin, origin + axis_length * x_axis, [1, 0, 0], lineWidth=2, lifeTime=0.1)
        p.addUserDebugLine(origin, origin + axis_length * y_axis, [0, 1, 0], lineWidth=2, lifeTime=0.1)
        p.addUserDebugLine(origin, origin + axis_length * z_axis, [0, 0, 1], lineWidth=2, lifeTime=0.1)

        view_flat = p.computeViewMatrix(cameraEyePosition=origin, cameraTargetPosition=origin-z_axis, cameraUpVector=y_axis)

        # # compute projection using FOV
        foc = params['focal_length']
        v_ap = params['v_ap']
        h_ap = params['h_ap']
        near, far = params['near'], params['far']
        # # vertical FOV in degrees
        # fov_y = math.degrees(2 * math.atan((v_ap/2) / foc))
        fov_h = math.degrees(2 * math.atan((h_ap/2) / foc))
        # fov_h =  2*math.atan((h_ap/2) / foc)
        fov_v = math.degrees(2*math.atan((h_ap/w*h/2) / foc))
        # fov_y = params['fov']
        aspect = w / h
        proj_mat = p.computeProjectionMatrixFOV(fov_v, aspect, near, far)

        # view_flat = view_mat.flatten(order='F').tolist()
        # proj_mat is already a flat list of 16 floats

        # Render images
        _, _, rgb, _, seg = p.getCameraImage(
            width=w,
            height=h,
            viewMatrix=view_flat,
            projectionMatrix=proj_mat,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        level1_uids = np.array([self.robot_uid] + self.moving_obj_uids)
        level2_uids = np.array(self.moving_obj_uids)
        segs = np.stack([np.any(seg[...,None] == level1_uids, axis=-1),
                        np.any(seg[...,None] == level2_uids, axis=-1)], axis=-1)

        # seg is a 2D array of instance ids
        return segs

    
    def set_robot_q(self, q):
        # Reset robot joint states
        if q.shape[-1] > p.getNumJoints(self.robot_uid):
            q_se2 = q[:3]
            base_pos = np.concatenate([q_se2[:2], [self.robot_height]])
            base_quat = scipy.spatial.transform.Rotation.from_euler('z', q_se2[2]).as_quat()
            p.resetBasePositionAndOrientation(self.robot_uid, base_pos, base_quat)
            q = q[3:]
        for idx, val in enumerate(q):
            p.resetJointState(self.robot_uid, idx+1, val)

        # Update each moving object pose based on end-effector
        for uid, link_idx, (ee_pos, ee_ori) in zip(
            self.moving_obj_uids,
            self.ee_link_indices,
            self.ee_to_obj_pqc
        ):
            link_state = p.getLinkState(
                self.robot_uid, link_idx, computeForwardKinematics=True
            )
            link_pos, link_ori = link_state[4], link_state[5]
            obj_pos, obj_ori = p.multiplyTransforms(
                link_pos, link_ori, ee_pos, ee_ori
            )
            p.resetBasePositionAndOrientation(uid, obj_pos, obj_ori)

        
def create_movement_figure(cache_dir, output_base_name: str):
    os.makedirs(output_base_name, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(cache_dir, "*.png")))[5:]
    seg_paths = sorted(glob.glob(os.path.join(cache_dir, "*.npy")))[5:]

    gaps = [len(image_paths)//10, len(image_paths)//20, len(image_paths)//30, len(image_paths)//40, 2]
    concat_imgs = [None]*len(gaps)
    img_cnts = [0]*len(gaps)
    opas = [0.7, 0.5, 0.3]
    for opa in opas:
        for i, (image_path, seg_path) in enumerate(zip(image_paths, seg_paths)):
            for gap_idx, gap in enumerate(gaps):
                if i % gap == 0:
                    image = Image.open(image_path)
                    rgb = np.array(image).astype(np.int32)
                    seg_load = np.load(seg_path)
                    if i==0:
                        concat_imgs[gap_idx] = copy.deepcopy(rgb)
                    else:
                        concat_imgs[gap_idx] = np.where(seg_load[...,0:1], rgb*opa + concat_imgs[gap_idx]*(1-opa), concat_imgs[gap_idx])
                    img_cnts[gap_idx] += 1
        
        # for i, (image_path, seg_path) in enumerate(zip(image_paths, seg_paths)):
        #     for gap_idx, gap in enumerate(gaps):
        #         if i % gap == 0:
        #             image = Image.open(image_path)
        #             rgb = np.array(image).astype(np.int32)
        #             seg_load = np.load(seg_path)
        #             concat_imgs[gap_idx] = np.where(seg_load[...,1:2], rgb*opa + concat_imgs[gap_idx]*(1-opa), concat_imgs[gap_idx])
        #             img_cnts[gap_idx] += 1
        
        for gap_idx, gap in enumerate(gaps):
            concat_imgs[gap_idx] = concat_imgs[gap_idx].astype(np.uint8)
            concat_imgs[gap_idx] = Image.fromarray(concat_imgs[gap_idx])
            concat_imgs[gap_idx].save(os.path.join(output_base_name, f"concat_{gaps[gap_idx]}_{opa}_imgs.png"))


class VideoWriter:
    def __init__(self, video_filename: str = None, cache_dir: str = None):
        self._frame_id = 0
        self.video_filename = video_filename
        self.cache_dir = cache_dir
        os.system(f"rm -r {self.cache_dir}")

        os.makedirs(self.cache_dir, exist_ok=True)
        self.frames = []


    def close(self):
        image_paths = sorted(glob.glob(os.path.join(self.cache_dir, "*.png")))[5:]
        image_paths = [image_path for image_path in image_paths if "background" not in image_path]
        background_image_path = os.path.join(self.cache_dir, "background.png")
        # background_image = Image.open(background_image_path)
        # image_filename = self.video_filename.replace(".mp4", ".png")

        clip = ImageSequenceClip(image_paths, fps=60)

        # frames = list(clip.iter_frames())
        # frames = np.stack(frames, axis=0)
        # background_frame = np.array(background_image)[:, :, :3]
        # robot_mask = np.abs((frames.astype(float) - background_frame[None].astype(float))).sum(axis=-1) > 100
        # robot_mask_count = robot_mask.sum(axis=0)
        # frames[~robot_mask] = 0
        # frame_mean = frames.sum(axis=0) / (robot_mask_count[:, :, None] + 1e-6)

        # # idx = robot_mask_count[:, :, None] == 0
        # # frame_mean[idx] = background_frame[idx]

        # ratio = 0.9
        # save_image = frame_mean * ratio + background_frame * (1 - ratio)
        # breakpoint()
        # mask = robot_mask_count == 0
        # save_image[mask] = background_frame[mask]
        # save_image = save_image.astype(np.uint8)
        # save_image = Image.fromarray(save_image)
        # save_image.save(image_filename)
        # print(f"Image saved to {image_filename}")

        clip.write_videofile(self.video_filename, audio=False)
        print(f"Video saved to {self.video_filename}")
        
        create_movement_figure(self.cache_dir, self.video_filename[:-4])

def load_yaml(file_path: str):
    import yaml
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_object_path(obj_path, project_dir):
    if obj_path == "":
        return None
    if "plane.obj" in obj_path:
        return None
    if "/dataset/object_set/" in obj_path:
        obj_path = obj_path.replace("/dataset/object_set/", "/home/rogga/research/efficient_planning/dataset/")
    if obj_path.startswith("assets"):
        obj_path = os.path.join(project_dir, obj_path)
    elif "pybullet_data" in obj_path:
        obj_file_name = os.path.basename(obj_path)
        pybullet_data_path = pybullet_data.getDataPath()
        obj_path = os.path.join(pybullet_data_path, obj_file_name)
    return obj_path

def load_meshes(mesh_dict, project_dir):
    np_rng = np.random.default_rng(2)
    prev_obj_path = None
    for key, value in mesh_dict.items():
        obj_path = value["file_path"]
        obj_path = get_object_path(obj_path, project_dir)
        if obj_path is None:
            continue

        mesh_scale = (value["scale"],) * 3
        mesh_pose = value["pose"]

        position = mesh_pose[:3]
        quaternion_scalar_first = mesh_pose[3:]
        # wxyz quaternion to rotation
        quaternion = np.concatenate([quaternion_scalar_first[1:], quaternion_scalar_first[:1]])
        rotation = scipy.spatial.transform.Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)
        # rotation = scipy.spatial.transform.Rotation.from_quat(quaternion, scalar_first=True).as_euler('xyz', degrees=True)
        position = Gf.Vec3d(position[0], position[1], position[2])
        rotation = Gf.Vec3d(rotation[0], rotation[1], rotation[2])

        # Load the mesh into the simulation
        prim_path = f"/World/{key}"
        add_reference_to_stage(obj_path, Sdf.Path(prim_path))

        # Set the scale, position, and rotation of the mesh
        mesh_prim = get_current_stage().GetPrimAtPath(prim_path)

        xformable = UsdGeom.Xformable(mesh_prim)
        xformable.SetXformOpOrder([])
        xformable.AddTranslateOp().Set(position)
        xformable.AddRotateXYZOp().Set(rotation)
        xformable.AddScaleOp().Set(mesh_scale)
        
        if prev_obj_path is not None and prev_obj_path==obj_path:
            np_rng = np_rng_prev
        np_rng_prev =copy.deepcopy(np_rng)            
        set_color(np_rng, mesh_prim)
        prev_obj_path = obj_path


def set_color(np_rng, mesh_prim):
    stage = get_current_stage()
    
    # 1) pick a random RGB color
    rand_color = Gf.Vec3f(np_rng.random(),
                        np_rng.random(),
                        np_rng.random())

    # 2) create & bind a new MDL material using Omni’s PBR library
    mtl_created = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=mtl_created,
    )
    mtl_prim = stage.GetPrimAtPath(mtl_created[0])

    # 3) feed the random color into its "baseColor" input
    omni.usd.create_material_input(
        mtl_prim,
        "diffuse_color_constant",                    # PBR's diffuse input
        rand_color,
        Sdf.ValueTypeNames.Color3f
    )

    # # 4) bind it *strongly* to your mesh prim
    mat = UsdShade.Material(mtl_prim)
    UsdShade.MaterialBindingAPI(mesh_prim).Bind(
        mat,
        UsdShade.Tokens.strongerThanDescendants
    )

def load_robot(robot, robot_prim_path, robot_config, project_dir):
    robot_base_pose = robot_config["robot_base_pose"]
    position = robot_base_pose[:3]
    quaternion_scalar_first = robot_base_pose[3:]
    quaternion = np.concatenate([quaternion_scalar_first[1:], quaternion_scalar_first[:1]])
    # wxyz quaternion to rotation
    # rotation = scipy.spatial.transform.Rotation.from_quat(quaternion, scalar_first=True).as_euler('xyz', degrees=True)
    rotation = scipy.spatial.transform.Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)
    position = Gf.Vec3d(position[0], position[1], position[2])
    rotation = Gf.Vec3d(rotation[0], rotation[1], rotation[2])

    robot.set_world_pose(
        position=position,
        orientation=quaternion_scalar_first,
    )
    # stage = omni.usd.get_context().get_stage()
    stage = get_current_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    link_prims = robot_prim.GetChildren()
    # geometry_prim = GeometryPrim(prim_paths_expr=robot_prim_path)
    # geometry_prim.disable_collision()

    np_rng = np.random.default_rng(0)

    # set_color(np_rng, robot_prim)
    # for lp in link_prims:
    #     set_color(np_rng, lp)

    moving_objects = robot_config["moving_obj"]
    grasping_objects = []
    for idx, value in enumerate(moving_objects):
        ee_to_obj_pq = value["ee_to_obj_pq"]
        attached_to = value["attached_to"]

        # we need to attach the object to the robot
        link_prim = link_prims[3 + attached_to]
        # link_pose = omni.usd.get_world_transform_matrix(link_prim, 0)

        ee_to_obj_h = Gf.Matrix4d()
        ee_to_obj_position = ee_to_obj_pq[:3]
        ee_to_obj_quaternion = ee_to_obj_pq[3:]

        ee_to_obj_h.SetTranslateOnly(Gf.Vec3d(ee_to_obj_position[0], ee_to_obj_position[1], ee_to_obj_position[2]))
        ee_to_obj_h.SetRotateOnly(Gf.Quatd(ee_to_obj_quaternion[0], ee_to_obj_quaternion[1], ee_to_obj_quaternion[2], ee_to_obj_quaternion[3]))

        key = f"moving_obj_{idx}"
        obj_path = value["file_path"]
        obj_path = get_object_path(obj_path, project_dir)
        if obj_path is None:
            continue

        obj_scale = (value["scale"],) * 3

        # Load the mesh into the simulation
        if obj_path.startswith("assets"):
            obj_path = os.path.join(project_dir, obj_path)
        prim_path = f"/World/{key}"
        add_reference_to_stage(obj_path, Sdf.Path(prim_path))

        # Set the scale, position, and rotation of the mesh
        mesh_prim = get_current_stage().GetPrimAtPath(prim_path)

        xformable = UsdGeom.Xformable(mesh_prim)
        xformable.SetXformOpOrder([])
        xformable.AddRotateXYZOp()
        xformable.AddTranslateOp()
        xformable.AddScaleOp().Set(obj_scale)

        set_color(np_rng, mesh_prim)

        grasping_objects.append({
            "mesh_prim": mesh_prim,
            "link_prim": link_prim,
            "ee_to_obj_h": ee_to_obj_h,
            "obj_scale": obj_scale,
            "prim_path": prim_path,
        })

    return grasping_objects

def evaluate_full_trajectory(control_points, samples_per_segment):
    """
    Evaluate the full trajectory by sampling each segment.
    
    control_points: array of shape (N, 6).
    samples_per_segment: number of sample points per segment.
    Returns: a concatenated array of trajectory values. ((N-1)*samples_per_segment, 6).
    """
    NT = control_points.shape[-2]
    # if NT == 2:
    #     control_points = way_points_to_trajectory(control_points, samples_per_segment, cos_transition=False)
    coeffs = broad_phase.SE3_interpolation_coeffs(control_points)
    return broad_phase.SE3_interpolation_eval(*coeffs, np.linspace(0, 1, (NT-1)*samples_per_segment+1, endpoint=True))

def decompose_matrix(mat: Gf.Matrix4d):
    reversed_ident_mtx = reversed(Gf.Matrix3d())

    translate = mat.ExtractTranslation()
    scale = Gf.Vec3d(*(v.GetLength() for v in mat.ExtractRotationMatrix()))
    #must remove scaling from mtx before calculating rotations
    mat.Orthonormalize()
    #without reversed this seems to return angles in ZYX order
    rotate = Gf.Vec3d(*reversed(mat.ExtractRotation().Decompose(*reversed_ident_mtx)))
    return translate, rotate, scale

def save_capture_params(param_filename):
    viewport_api = get_active_viewport()
    camera_path = viewport_api.camera_path.pathString
    stage = omni.usd.get_context().get_stage()
    camera_prim = stage.GetPrimAtPath(camera_path)
    camera = UsdGeom.Camera(camera_prim)
    gf_camera = camera.GetCamera()

    # Resolution
    width, height = viewport_api.resolution

    # Get camera transform (camera to world)
    cam_to_world = np.array(gf_camera.transform).reshape(4, 4).astype(np.float32)
    # Get intrinsic parameters
    focal_length = gf_camera.focalLength
    horizontal_aperture = camera.GetHorizontalApertureAttr().Get()
    vertical_aperture = camera.GetVerticalApertureAttr().Get()
    near, far = camera.GetClippingRangeAttr().Get()

    # Store camera parameters for PyBullet
    cam_params = {
        'width': width,
        'height': height,
        'cam_transform': gf_camera.transform,
        'cam_to_world': cam_to_world,
        'focal_length': focal_length,
        'h_ap': horizontal_aperture,
        'v_ap': vertical_aperture,
        'near': near,
        'far': far
    }

    with open(param_filename, 'wb') as f:
        pickle.dump(cam_params, f)

def visualize(config_path: str, video_filename: str, adjust_camera=True):

    config = load_yaml(config_path)

    exp_type = video_filename.split("/")[-3]

    world = World(stage_units_in_meters=1.0)

    # if "construction_site" in config_path:
    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()
    if 'construction_site' in exp_type:
        for i, ypos in enumerate(np.linspace(-5, 6, 6, endpoint=True)):
            sphere_light = UsdLux.SphereLight.Define(stage, Sdf.Path(f"/SphereLight{i}"))
            sphere_light.CreateIntensityAttr(300000)
            xformable1 = UsdGeom.Xformable(sphere_light)
            xformable1.AddTranslateOp().Set(Gf.Vec3d(0, ypos, 1.7))
            xformable1.AddScaleOp().Set(Gf.Vec3d(0.1, 0.1, 0.1))

        # sphere_light2 = UsdLux.SphereLight.Define(stage, Sdf.Path("/SphereLight2"))
        # sphere_light2.CreateIntensityAttr(1000)
        # xformable2 = UsdGeom.Xformable(sphere_light2)
        # xformable2.AddTranslateOp().Set(Gf.Vec3d(0, -6, 1.7))
    else:
        distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        distantLight.CreateIntensityAttr(1000)


    urdf_interface = _urdf.acquire_urdf_interface()
    import_config = _urdf.ImportConfig()
    import_config.convex_decomp = False  # Disable convex decomposition for simplicity
    import_config.fix_base = True       # Fix the base of the robot to the ground
    import_config.make_default_prim = True  # Make the robot the default prim in the scene
    import_config.self_collision = False  # Disable self-collision for performance
    import_config.distance_scale = 1     # Set distance scale for the robot
    import_config.density = 0.0          # Set density to 0 (use default values)

    # project_dir = "/home/rogga/research/efficient_planning/PointObjRep"
    project_dir = "/home/dongwon/research/PointObjRep"
    robot_urdf_path = os.path.join(project_dir, config["robot"]["urdf_path"])

    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile",
        urdf_path=robot_urdf_path,
        import_config=import_config
    )
    result, robot_prim_path = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_robot=robot_model,
        import_config=import_config,
    )
    robot = Robot(prim_path=robot_prim_path)
    load_meshes(config["mesh"], project_dir)
    plane_path = "/World/defaultGroundPlane"
    # adjust plane translation
    plane_prim = get_current_stage().GetPrimAtPath(plane_path)
    if "bimanual" in config_path:
        plane_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, -0.2))
    if "dish" in config_path:
        plane_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, -0.3))
    

    # turn off the gravity
    scene_path = Sdf.Path("/World/PhysicsScene")
    scene = UsdPhysics.Scene.Get(stage, scene_path)
    if not scene:
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene = UsdPhysics.Scene.Get(stage, prim.GetPath())
                break
    if not scene:
        raise RuntimeError("No UsdPhysics.Scene prim found in the stage")
    scene.GetGravityDirectionAttr().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    scene.GetGravityMagnitudeAttr().Set(0.0)

    world.play()
    robot.initialize()
    grasping_objects = load_robot(robot, robot_prim_path, config["robot"], project_dir)

    qs = np.array(config["robot"]["traj"])
    nintp = qs.shape[-2]
    interpolated_qs = evaluate_full_trajectory(qs, samples_per_segment=20)[0]
    i = 0

    viewport_api, viewport_window  = get_active_viewport_and_window()

    # camera_path = viewport_api.camera_path.pathString
    # stage = omni.usd.get_context().get_stage()
    # camera_prim = stage.GetPrimAtPath(camera_path)
    # camera = UsdGeom.Camera(camera_prim)
    
    # 4. Apply the transform to the USD camera prim
    # xformable = UsdGeom.Xformable(camera_prim)
    # clear any existing xform ops
    # xformable.SetXformOpOrder([])
    # xformable.GetXformOpOrderAttr().Set([])
   
    world.step(render=True)

    writer = VideoWriter(video_filename=video_filename, cache_dir="_temp")

    for _ in range(50):
        simulation_app.update()
    

    if os.path.exists('cam_params.pkl'):
        with open('cam_params.pkl', 'rb') as f:
            cam_param = pickle.load(f)

        cam_to_world = cam_param['cam_to_world']
        cam2world = cam_to_world.T
        origin = cam2world[:3, 3]
        z_axis = cam2world[:3, 2]

        set_camera_view(eye=origin, target=origin-z_axis)

        for _ in range(10):
            world.step(render=True)
        
    if adjust_camera:
        while world.is_playing():
            world.step(render=True)
        while not world.is_playing():
            world.step(render=True)
    warn_first = True

    # save camera transform
    save_capture_params('cam_params.pkl')

    # if os.path.exists('cam_params.pkl'):
    #     with open('cam_params1.pkl', 'rb') as f:
    #         cam_param1 = pickle.load(f)

    #     cam_to_world = cam_param1['cam_to_world']
    #     cam2world = cam_to_world.T
    #     origin = cam2world[:3, 3]
    #     x_axis = cam2world[:3, 0]
    #     y_axis = cam2world[:3, 1]
    #     z_axis = cam2world[:3, 2]

    #     set_camera_view(eye=origin, target=origin-z_axis)

    # quat_ori = scipy.spatial.transform.Rotation.from_matrix(cam2world[:3,:3]).as_quat()
    # quat_ori_scalar_first = np.concatenate([quat_ori[3:], quat_ori[:3]])
    # camera = Camera(
    #     prim_path="/World/camera",
    #     position=origin,
    #     frequency=20,
    #     resolution=(256, 256),
    #     orientation=quat_ori_scalar_first,
    # )
    # camera.initialize()

    # capture_from_custom_camera(cam_param1)

    pb_scene = PBSCene(config)

    # world.pause()
    interpolated_qs = np.array(interpolated_qs)
    start_end_nrepeat = (len(interpolated_qs)*0.1)
    interpolated_qs = np.concatenate([interpolated_qs[:1].repeat(start_end_nrepeat, 0), 
                                      interpolated_qs, interpolated_qs[-1:].repeat(start_end_nrepeat, 0)], axis=0)
    copied_robots = []
    while True:
        if world.is_playing():
            warn_first = True

            # q = interpolated_qs[i % len(interpolated_qs)]
            cur_step = min(i, len(interpolated_qs)-1)
            q = interpolated_qs[cur_step]
            # pb_scene.set_robot_q(interpolated_qs[max(cur_step-1, 0)])
            pb_scene.set_robot_q(interpolated_qs[cur_step])
            i += 1
            se2 = None
            if len(q) == 12: # bimanual
                q = [q[0], q[6], q[1], q[7],  q[2], q[8], q[3], q[9], q[4], q[10], q[5], q[11]]
            elif len(q) == 9: # construction site
                se2 = q[0:3]
                q = q[3:]
            if se2 is not None:
                world_pose = robot.get_world_pose()
                position = Gf.Vec3d(float(se2[0]), float(se2[1]), float(world_pose[0][2]))
                quaternion = scipy.spatial.transform.Rotation.from_euler('z', se2[2], degrees=False).as_quat()
                quaternion_scalar_first = np.concatenate([quaternion[...,3:], quaternion[...,:3]], axis=-1)
                robot.set_world_pose(
                    position=position,
                    orientation=quaternion_scalar_first,
                )

            for _ in range(10):
                robot.set_joint_positions(q)
                # world.step(render=True)

                for grasping_object in grasping_objects:
                    link_prim = grasping_object["link_prim"]
                    link_to_obj_h = grasping_object["ee_to_obj_h"]
                    grasping_mesh_prim = grasping_object["mesh_prim"]

                    timeline = omni.timeline.get_timeline_interface()
                    timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()
                    link_pose = omni.usd.get_world_transform_matrix(link_prim, timecode)

                    obj_pose = link_to_obj_h * link_pose
                    obj_position, obj_rotation, _ = decompose_matrix(obj_pose)

                    xformable = UsdGeom.Xformable(grasping_mesh_prim)
                    xformable.SetXformOpOrder([])
                    xformable.AddTranslateOp().Set(obj_position)
                    xformable.AddRotateXYZOp().Set(obj_rotation)
                    xformable.AddScaleOp().Set(grasping_object["obj_scale"])

                world.step(render=True)
            world.step(render=True)
            capture_helper = capture_viewport_to_file(viewport_api, os.path.join(writer.cache_dir, f"{i:04d}.png"))

            # res = camera.get_current_frame()

            seg = pb_scene.get_imgs()
            # save to tmp np
            seg_filepath = os.path.join(writer.cache_dir, f"{i:04d}_seg")
            np.save(seg_filepath, np.array(seg))

            # # # debug visualization
            # test_idx = 40
            # image_path = os.path.join(writer.cache_dir, f"{test_idx:04d}.png")
            # if os.path.exists(image_path):
            #     image = Image.open(image_path)
            #     rgb = np.array(image)
            #     seg_load = np.load(os.path.join(writer.cache_dir, f"{test_idx:04d}_seg.npy"))
            #     segmented_rgb = np.where(seg_load[...,0:1], rgb, 0)
            #     import matplotlib.pyplot as plt
            #     plt.figure()
            #     plt.subplot(3,1,1)
            #     plt.imshow(rgb)
            #     plt.subplot(3,1,2)
            #     plt.imshow(seg_load[...,0])
            #     plt.subplot(3,1,3)
            #     plt.imshow(segmented_rgb)
            #     plt.show()


        else:
            if warn_first:
                print("Simulation is paused. Press 'Play' to continue.")
                warn_first = False
            world.step(render=True)
            capture_viewport_to_file(viewport_api, os.path.join(writer.cache_dir, f"{i:04d}.png"))

        if i % len(interpolated_qs) == 0:
        # if i >= int(len(interpolated_qs)*1.1):
            print("====================")
            break
    
    time.sleep(1)
    writer.close()
    world.clear()

if __name__ == "__main__":

    app = omni.kit.app.get_app_interface()
    app.get_extension_manager().set_extension_enabled("omni.syntheticdata", True)

    import glob
    base_dir = "/home/dongwon/research/PointObjRep/temp"
    # video_folder = "video/dish"
    # seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # video_folder = "video/bimanual"
    # seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # video_folder = "video/construction_site"
    # seeds = [0, 12, 18, 19, 22, 23, 24, 25, 29, 30]

    # ccd_type = "curobo"
    # video_folder = "video/dish"
    # seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # ccd_type = "ours"
    # video_folder = "video/construction_site_hard"
    # seeds = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13]

    ccd_type = "ours"
    video_folder = "video/dish_multiple"
    seeds = [0, 1, 2, 3, 4, 5, 6] # 6]
    # seeds = [[seed * 4 + i for i in range(4)] for seed in seeds]
    # seeds = [seed for sublist in seeds for seed in sublist]
    # print(seeds)

    # ccd_type = "ours"
    # video_folder = "video/bimanual_hard"
    # seeds = [0,1,2,34, 5, 6,7,8,9,10,11,12,13]

    os.makedirs(video_folder, exist_ok=True)
    config_paths = [
        os.path.join(base_dir, video_folder, ccd_type, f"{i}.yml")
        for i in seeds
    ]
    for i, config_path in enumerate(config_paths):
        video_filename = os.path.join(video_folder, ccd_type, f"{seeds[i]}.mp4")
        os.makedirs(os.path.dirname(video_filename), exist_ok=True)
        print(f"Visualizing {config_path} to {video_filename}")
        visualize(config_path, video_filename, adjust_camera=(i==0))
        
    # config_path = "/home/rogga/research/efficient_planning/PointObjRep/temp/video/bimanual/ours/0.yml"
    # visualize(config_path, video_filename, camera_info)
 
    # config_path = "/home/rogga/research/efficient_planning/PointObjRep/temp/video/construction_site/ours/0.yml"
    # visualize(config_path, video_filename)

    simulation_app.close()
