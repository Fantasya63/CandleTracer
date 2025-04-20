from Scene.Scene import Scene
from Renderer.Film import Film
from Scene.RTPrimitives import Sphere, Cube, Plane
from numba import cuda
from numba import void, float32, int32, uint8

import numpy as np
import math


@cuda.jit(device=True)
def ray_sphere_intersect(ray_origin, ray_dir, sphere_pos, sphere_radius):
    oc = ray_origin - sphere_pos
    a = ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2
    b = 2.0 * (oc[0] * ray_dir[0] + oc[1] * ray_dir[1] + oc[2] * ray_dir[2])
    c = oc[0]**2 + oc[1]**2 + oc[2]**2 - sphere_radius**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return -1.0
    t = (-b - math.sqrt(discriminant)) / (2.0 * a)
    return t if t > 0 else -1.0


@cuda.jit(device=True)
def ray_cube_intersect(ray_origin, ray_dir, cube_pos, cube_size):
    inv_dir = cuda.local.array(3, dtype=np.float32)
    for i in range(3):
        inv_dir[i] = 1.0 / ray_dir[i] if ray_dir[i] != 0 else math.inf
    
    t_min = -math.inf
    t_max = math.inf
    
    for i in range(3):
        t1 = (cube_pos[i] - cube_size[i] - ray_origin[i]) * inv_dir[i]
        t2 = (cube_pos[i] + cube_size[i] - ray_origin[i]) * inv_dir[i]
        t_min = max(t_min, min(t1, t2))
        t_max = min(t_max, max(t1, t2))
    
    if t_min > t_max or t_max < 0:
        return -1.0
    return t_min if t_min > 0 else t_max if t_max > 0 else -1.0


@cuda.jit(device=True)
def ray_plane_intersect(ray_origin, ray_dir, plane_normal, plane_distance):
    denom = plane_normal[0] * ray_dir[0] + plane_normal[1] * ray_dir[1] + plane_normal[2] * ray_dir[2]
    if abs(denom) < 1e-6:
        return -1.0
    t = -(plane_normal[0] * ray_origin[0] + plane_normal[1] * ray_origin[1] + 
          plane_normal[2] * ray_origin[2] + plane_distance) / denom
    return t if t > 0 else -1.0

@cuda.jit
def Trace(outData, viewProj, camPos, spherePos, sphereRadius, numSpheres, width, height):
    
    x, y = cuda.grid(2)
    
    # Check for bounds
    if x >= width or y >= height:
        return
    
    ndcX = (x / width) * 2.0 - 1.0
    ndcY = (y / height) * 2.0 - 1.0

    ray_dir = cuda.local.array(3, dtype=np.float32)
    ray_dir[0] = viewProj[0, 0] * ndcX + viewProj[0, 1] * ndcY + viewProj[0, 2]
    ray_dir[1] = viewProj[1, 0] * ndcX + viewProj[1, 1] * ndcY + viewProj[1, 2]
    ray_dir[2] = viewProj[2, 0] * ndcX + viewProj[2, 1] * ndcY + viewProj[2, 2]

    # Normalize
    mag = math.sqrt(ray_dir[0]**2 + ray_dir[1]**2 + ray_dir[2]**2)
    for i in range(3):
        ray_dir[i] /= mag


    closestDist = math.inf
    hitPos = cuda.local.array(3, dtype=np.float32)
    hitPos[0] = 0.0
    hitPos[1] = 0.0
    hitPos[2] = 0.0
    
    for i in range(numSpheres):
        t = -1.0
        t = ray_sphere_intersect(ray_origin=camPos, ray_dir=ray_dir, sphere_pos=spherePos[i], sphere_radius=sphereRadius[i])

        if t > 0.0 and t < closestDist:
            closestDist = t

    if closestDist != math.inf:
        for c in range(3):
            outData[y, x, c] = closestDist
    else:
        for c in range(3):
            outData[y, x, c] = 0.0




class Renderer:
    def __init__(self):
        pass

    def Render(self, scene : Scene, film : Film):
        
        numSphere = len(scene.spheres)

        camera = scene.camera
        cudaCamPos = cuda.to_device(camera.position)
        cudaViewProj = cuda.to_device(camera.viewProjection)


        spherePos = np.array([sphere.position for sphere in scene.spheres])
        sphereRad = np.array([sphere.radius for sphere in scene.spheres])
        # Serialize Scene data to GPU
        # for i in range(numSphere):
        #     sphere : Sphere = scene.spheres[i]
        #     spherePos[i] = sphere.position
        #     sphereRad[i] = sphere.radius

        cudaSpherePos = cuda.to_device(spherePos)
        cudaSphereRad = cuda.to_device(sphereRad)
        cudaOutData = cuda.to_device(film.data)

        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(scene.camera.film.width / threads_per_block[0])
        blocks_per_grid_y = math.ceil(scene.camera.film.height / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        Trace[blocks_per_grid, threads_per_block](cudaOutData, cudaViewProj, cudaCamPos, cudaSpherePos, cudaSphereRad, numSphere, film.width, film.height)

        scene.camera.film.pixels = cudaOutData.copy_to_host()
        return scene.camera.film.pixels