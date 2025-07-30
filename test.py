import tal
import time
import numpy as np
import matplotlib.pyplot as plt

data = tal.io.read_capture('C:/Datasets/transient/nlos/z/confocal-scene-64.hdf5')  # you'll need to generate this file using "tal render exhaustive-scene"
tal.reconstruct.compensate_laser_cos_dsqr(data)

#volume_xyz = tal.reconstruct.get_volume_project_rw(data, depths=[1.0,])

# ------------------------------------------
start_time = time.time()

# ------ Backprojection -------------
# with tal.resources(cpu_processes='max', downscale=1):
#     H_1 = tal.reconstruct.bp.solve(data, volume_xyz=volume_xyz, camera_system=tal.enums.CameraSystem.DIRECT_LIGHT)

# ------ Filtered backprojection -------------
# with tal.resources(cpu_processes='max', downscale=1):
#     H_1 = tal.reconstruct.fbp.solve(data, volume_xyz=volume_xyz, camera_system=tal.enums.CameraSystem.DIRECT_LIGHT, wl_mean=0.25, wl_sigma=0)

with tal.resources(cpu_processes='max', downscale=1):
    H_1 = tal.reconstruct.fk.solve(data)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Measured time: {elapsed_time:.4f} seconds")

# -------------------------------
tal.plot.txy_interactive(H_1)

max_z = np.max(H_1, axis=0)  # Take the maximum projection across the time dimension
plt.imshow(max_z, cmap='magma')
plt.title('Reconstruction')
plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()