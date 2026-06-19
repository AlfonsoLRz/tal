import tal
import time
import numpy as np
import matplotlib.pyplot as plt

#data = tal.io.read_capture(
#    'C:/Datasets/transient/nlos/bunny/exhaustive-scene-32.hdf5'
#)
# data = tal.io.read_capture(
#     'nlos-z.hdf5'
# )
data = tal.io.read_capture(
    'C:/Datasets/transient/droyo/v2-officescene/officescene/rect/officescene.hdf5'
)
tal.reconstruct.compensate_laser_cos_dsqr(data)

depths = np.linspace(0.5, 4.0, 100)
volume_xyz = tal.reconstruct.get_volume_project_rw(data, depths=depths)
print(volume_xyz.shape)

# volume_max = np.array([1.0, 1.0, 2.0])
# volume_min = np.array([-1.0, -1.0, 0.5])
# volume_xyz = tal.reconstruct.get_volume_min_max_resolution(maximal_pos=volume_max, minimal_pos=volume_min, resolution=0.02)

# volume_min = np.array([-1.0, -0.7, 0.5])
# volume_max = np.array([ 1.0,  0.7, 2.0])
# volume_xyz = tal.reconstruct.get_volume_min_max_resolution(minimal_pos=volume_min, maximal_pos=volume_max, resolution=0.02,)

# ------------------------------------------
start_time = time.time()

# ------ Backprojection -------------
# with tal.resources(cpu_processes='max', downscale=1):
#    H_1 = tal.reconstruct.bp.solve(data, volume_xyz=volume_xyz, camera_system=tal.enums.CameraSystem.DIRECT_LIGHT)

# ------ Filtered backprojection -------------
# with tal.resources(cpu_processes='max', downscale=1):
#     H_1 = tal.reconstruct.fbp.solve(data, volume_xyz=volume_xyz, camera_system=tal.enums.CameraSystem.DIRECT_LIGHT, wl_mean=0.05, wl_sigma=0.05)

# H_1 = tal.reconstruct.fk.solve(data, downscale=1,)

# V = np.moveaxis(np.mgrid[-1:1.1:0.1, -1:1.1:0.1, 0.5:2.6:0.1], 0, -1).reshape(-1,3)
# # Reconstruct the data to the volume V with virtual illumination pulse
# # with central wavefactor 6 and 4 cycles
H_1 = tal.reconstruct.pf.solve(data, 0.06, 4, camera_system=tal.enums.CameraSystem.DIRECT_LIGHT, volume=volume_xyz)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Measured time: {elapsed_time:.4f} seconds")

# -------------------------------
#tal.plot.txy_interactive(H_1)

max_z = np.max(np.abs(H_1), axis=2)  # Take the maximum projection across the depth dimension
max_z = np.rot90(max_z, 3)
plt.imshow(max_z, cmap='magma')
plt.title('Reconstruction')
plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
