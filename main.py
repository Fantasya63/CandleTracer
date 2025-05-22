from Log.Logger import *
# from Networking.client import client
# from Networking.server import server
from Scene.Scene import Scene
from Scene.SceneSerializer import SceneSerializer
from Renderer.Camera import Camera
from Renderer.Film import Film
from Scene.RTPrimitives import Sphere, Cube, Plane
from Renderer.Renderer import Renderer


import matplotlib.pyplot as plt

def main():
    # # setup
    # LogInfo("Distributed Offline Ray Tracing of CG Scenes")
    # LogInfo("What configuration would you like for this device?")
    # LogInfo("  s - server")
    # LogInfo("  c - client")
    # config = input("Enter your input: ")
    # config = config.lower()

    # if config == 's':
    #     pass
    # elif config == 'c':
    #     pass
    # else:
    #     LogError("Unknown input is encountered. Auto exiting...")
    #     return
    


    # Test
    scene : Scene = Scene()
    film : Film = Film(1920, 1080)
    camera : Camera = Camera(pos=[0.0, 0.0, 0.0], rot=[0.0, 0.0, 0.0], fov=90.0, near=0.1, far=10.0, film=film)
    scene.camera = camera
    sphere : Sphere = Sphere(position=[1.5, 0.0, -1.0], radius=0.5)
    scene.add_sphere(sphere)

    renderer : Renderer = Renderer()
    # result = renderer.Render(scene, film)
    result = renderer.TestGradient()

    plt.imshow(result)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()



