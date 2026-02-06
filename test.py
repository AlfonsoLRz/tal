import tal
import time
import numpy as np
import matplotlib.pyplot as plt

#data = tal.io.read_capture(
#    'C:/Datasets/transient/nlos/bunny/exhaustive-scene-32.hdf5'
#)  
data = tal.io.read_capture(
    'C:/Datasets/transient/nlos/z/confocal-scene-256.hdf5'
)  
tal.reconstruct.compensate_laser_cos_dsqr(data)

depths = np.linspace(0.5, 2.0, 30)
volume_xyz = tal.reconstruct.get_volume_project_rw(data, depths=depths)

# ------------------------------------------
start_time = time.time()

# ------ Backprojection -------------
#with tal.resources(cpu_processes='max', downscale=1):
#    H_1 = tal.reconstruct.bp.solve(data, volume_xyz=volume_xyz, camera_system=tal.enums.CameraSystem.DIRECT_LIGHT)

# ------ Filtered backprojection -------------
with tal.resources(cpu_processes='max', downscale=1):
    H_1 = tal.reconstruct.fbp.solve(data, volume_xyz=volume_xyz, camera_system=tal.enums.CameraSystem.DIRECT_LIGHT, wl_mean=0.25, wl_sigma=0.25 / np.sqrt(2))

#H_1 = tal.reconstruct.fk.solve(data, downscale=1,)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Measured time: {elapsed_time:.4f} seconds")

# -------------------------------
#tal.plot.txy_interactive(H_1)

max_z = np.max(H_1, axis=2)  # Take the maximum projection across the time dimension
plt.imshow(max_z, cmap='magma')
plt.title('Reconstruction')
plt.colorbar()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()