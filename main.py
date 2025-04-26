from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import time
import numpy as np
import scipy
import pybullet_data

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
from omni.kit.viewport.utility import get_active_viewport
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, Writer
import omni.usd
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

import carb.settings

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from PIL import Image

import broad_phase

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
    for key, value in mesh_dict.items():
        obj_path = value["file_path"]
        obj_path = get_object_path(obj_path, project_dir)
        if obj_path is None:
            continue

        mesh_scale = (value["scale"],) * 3
        mesh_pose = value["pose"]

        position = mesh_pose[:3]
        quaternion = mesh_pose[3:]
        # wxyz quaternion to rotation
        rotation = scipy.spatial.transform.Rotation.from_quat(quaternion, scalar_first=True).as_euler('xyz', degrees=True)
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
        
        # geometry_prim = GeometryPrim(prim_paths_expr=prim_path)
        # geometry_prim.disable_collision()

        # mesh_prim.GetAttribute("xformOp:scale").Set(mesh_scale)
        # mesh_prim.GetAttribute("xformOp:translate").Set(position)
        # if not mesh_prim.GetAttribute("xformOp:rotateXYZ"):
        #     UsdGeom.Xformable(mesh_prim).AddRotateXYZOp()
        # mesh_prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)

def load_robot(robot, robot_prim_path, robot_config, project_dir):
    robot_base_pose = robot_config["robot_base_pose"]
    position = robot_base_pose[:3]
    quaternion = robot_base_pose[3:]
    # wxyz quaternion to rotation
    rotation = scipy.spatial.transform.Rotation.from_quat(quaternion, scalar_first=True).as_euler('xyz', degrees=True)
    position = Gf.Vec3d(position[0], position[1], position[2])
    rotation = Gf.Vec3d(rotation[0], rotation[1], rotation[2])

    robot.set_world_pose(
        position=position,
        orientation=quaternion,
    )
    robot_prim = get_current_stage().GetPrimAtPath(robot_prim_path)
    link_prims = robot_prim.GetChildren()
    # geometry_prim = GeometryPrim(prim_paths_expr=robot_prim_path)
    # geometry_prim.disable_collision()

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

        # obj_pose = value["pose"]

        # position = obj_pose[:3]
        # quaternion = obj_pose[3:]

        # # wxyz quaternion to rotation
        # rotation = scipy.spatial.transform.Rotation.from_quat(quaternion, scalar_first=True).as_euler('xyz', degrees=True)
        # position = Gf.Vec3d(position[0], position[1], position[2])
        # rotation = Gf.Vec3d(rotation[0], rotation[1], rotation[2])

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

        # geometry_prim = GeometryPrim(prim_paths_expr=prim_path)
        # geometry_prim.disable_collision()

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

def visualize(config_path: str, video_filename: str, adjust_camera=True):

    config = load_yaml(config_path)
    world = World(stage_units_in_meters=1.0)

    # if "construction_site" in config_path:
    world.scene.add_default_ground_plane()


    stage = omni.usd.get_context().get_stage()
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

    project_dir = "/home/rogga/research/efficient_planning/PointObjRep"
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
    # arm = add_robot(
    #     prim_path="/World/Arm",
    #     asset_path = usd_path,
    # )
    load_meshes(config["mesh"], project_dir)
    plane_path = "/World/defaultGroundPlane"
    # adjust plane translation
    plane_prim = get_current_stage().GetPrimAtPath(plane_path)
    if "bimanual" in config_path or "dish" in config_path:
        plane_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, -0.5))

    world.play()
    robot.initialize()
    grasping_objects = load_robot(robot, robot_prim_path, config["robot"], project_dir)

    qs = np.array(config["robot"]["traj"])
    nintp = qs.shape[-2]
    interpolated_qs = evaluate_full_trajectory(qs, samples_per_segment=20)[0]
    i = 0

    viewport_api = get_active_viewport()
    action_registry = omni.kit.actions.core.get_action_registry()
    action = action_registry.get_action("omni.kit.viewport.actions", "toggle_grid_visibility")
    action.execute(viewport_api=viewport_api, visible=False)

    # obj_path = "/home/rogga/research/efficient_planning/PointObjRep/temp/video/construction_site_hard/stamp/13.obj"
    # add_reference_to_stage(obj_path, Sdf.Path(f"/World/swept_volume"))

    writer = VideoWriter(video_filename=video_filename, cache_dir="_temp")

    for _ in range(50):
        simulation_app.update()

    if adjust_camera:
        while world.is_playing():
            world.step(render=True)
        while not world.is_playing():
            world.step(render=True)
    warn_first = True
    # world.pause()
    copied_robots = []
    while True:
        if world.is_playing():
            warn_first = True

            q = interpolated_qs[i % len(interpolated_qs)]
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
                quaternion = scipy.spatial.transform.Rotation.from_euler('z', se2[2], degrees=False).as_quat(scalar_first=True)
                robot.set_world_pose(
                    position=position,
                    orientation=quaternion,
                )

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
                # obj_quaternion = scipy.spatial.transform.Rotation.from_euler('xyz', obj_rotation, degrees=True).as_quat(scalar_first=False)
                # print(obj_position, obj_rotation, obj_quaternion) #, obj_rotation)

                xformable = UsdGeom.Xformable(grasping_mesh_prim)
                xformable.SetXformOpOrder([])
                xformable.AddTranslateOp().Set(obj_position)
                xformable.AddRotateXYZOp().Set(obj_rotation)
                xformable.AddScaleOp().Set(grasping_object["obj_scale"])

            # world.step(render=True)

            # omni.kit.commands.execute("CopyPrims", paths_from=[robot_prim_path, grasping_object["prim_path"]], paths_to=[robot_prim_path + f"_{i}", grasping_object["prim_path"] + f"_{i}"], duplicate_layers=False, combine_layers=False)
            # copied_robot_prim = get_current_stage().GetPrimAtPath(robot_prim_path + f"_{i}")
            # copied_moving_obj_prim = get_current_stage().GetPrimAtPath(grasping_object["prim_path"] + f"_{i}")

            # copied_robots.append({
            #     "robot_prim_path": robot_prim_path + f"_{i}",
            #     "robot_q": q,
            #     "world_pose": (position, quaternion),
            #     "moving_obj_prim_path": grasping_object["prim_path"] + f"_{i}",
            #     "object_pose": (obj_position, obj_rotation, grasping_object["obj_scale"]),
            # })
            
            # for copied_robot in copied_robots:
            #     robot_prim_path = copied_robot["robot_prim_path"]
            #     c_robot = Robot(prim_path=copied_robot["robot_prim_path"])
            #     c_robot.set_joint_positions(copied_robot["robot_q"])
            #     c_robot.set_world_pose(
            #         position=position,
            #         orientation=quaternion,
            #     )

            #     object_prim = UsdGeom.Xformable(get_current_stage().GetPrimAtPath(copied_robot["moving_obj_prim_path"]))
            #     object_prim.SetXformOpOrder([])
            #     object_prim.AddTranslateOp().Set(copied_robot["object_pose"][0])
            #     object_prim.AddRotateXYZOp().Set(copied_robot["object_pose"][1])
            #     object_prim.AddScaleOp().Set(copied_robot["object_pose"][2])

            world.step(render=True)
            capture_viewport_to_file(viewport_api, os.path.join(writer.cache_dir, f"{i:04d}.png"))

        else:
            if warn_first:
                print("Simulation is paused. Press 'Play' to continue.")
                warn_first = False
            world.step(render=True)
            capture_viewport_to_file(viewport_api, os.path.join(writer.cache_dir, f"{i:04d}.png"))

        if i % len(interpolated_qs) == 0:
            print("====================")
            break
    
    # cnt = 0
    # while cnt < 15:
    #     # remove robot and get background
    #     robot.set_world_pose(
    #         position = Gf.Vec3d(100.0, 100.0, 100.0),
    #     )
    #     for grasping_object in grasping_objects:
    #         link_prim = grasping_object["link_prim"]
    #         link_to_obj_h = grasping_object["ee_to_obj_h"]
    #         grasping_mesh_prim = grasping_object["mesh_prim"]

    #         timeline = omni.timeline.get_timeline_interface()
    #         timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()
    #         link_pose = omni.usd.get_world_transform_matrix(link_prim, timecode)

    #         obj_pose = link_to_obj_h * link_pose
    #         obj_position, obj_rotation, _ = decompose_matrix(obj_pose)

    #         xformable = UsdGeom.Xformable(grasping_mesh_prim)
    #         xformable.SetXformOpOrder([])
    #         xformable.AddTranslateOp().Set(obj_position)
    #         xformable.AddRotateXYZOp().Set(obj_rotation)
    #         xformable.AddScaleOp().Set(grasping_object["obj_scale"])

    #     world.step(render=True)
    #     capture_viewport_to_file(viewport_api, os.path.join(writer.cache_dir, f"background.png"))

    #     cnt += 1
    time.sleep(1)
    writer.close()
    world.clear()

if __name__ == "__main__":
    import glob
    base_dir = "/home/rogga/research/efficient_planning/PointObjRep/temp"
    ccd_type = "ours"
    video_folder = "video/dish"
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    video_folder = "video/bimanual"
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    video_folder = "video/construction_site"
    seeds = [0, 12, 18, 19, 22, 23, 24, 25, 29, 30]

    ccd_type = "curobo"
    video_folder = "video/dish"
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    ccd_type = "ours"
    video_folder = "video/construction_site_hard"
    seeds = [0, 1, 2, 4, 5, 6]

    ccd_type = "ours"
    video_folder = "video/dish_multiple"
    seeds = [0, 1, 2, ] # 6]
    seeds = [[seed * 4 + i for i in range(4)] for seed in seeds]
    seeds = [seed for sublist in seeds for seed in sublist]
    print(seeds)

    ccd_type = "ours"
    video_folder = "video/bimanual_hard"
    seeds = [4, 5, 6]

    os.makedirs(video_folder, exist_ok=True)
    config_paths = [
        os.path.join(base_dir, video_folder, ccd_type, f"{i}.yml")
        for i in seeds
    ]
    for i, config_path in enumerate(config_paths):
        video_filename = os.path.join(video_folder, ccd_type, f"{seeds[i]}.mp4")
        visualize(config_path, video_filename, adjust_camera=(i==0))
        
    # config_path = "/home/rogga/research/efficient_planning/PointObjRep/temp/video/bimanual/ours/0.yml"
    # visualize(config_path, video_filename, camera_info)
 
    # config_path = "/home/rogga/research/efficient_planning/PointObjRep/temp/video/construction_site/ours/0.yml"
    # visualize(config_path, video_filename)

    simulation_app.close()
