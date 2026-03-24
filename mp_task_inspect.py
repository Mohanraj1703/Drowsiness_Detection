from mediapipe.tasks.python import vision, core
print('FaceLandmarker class methods:')
print([m for m in dir(vision.FaceLandmarker) if not m.startswith('_') and 'create' in m.lower()])
print('FaceLandmarkerOptions:', vision.FaceLandmarkerOptions)
print('image lib available:', [m for m in dir(vision) if 'image' in m.lower()])
from mediapipe.framework.formats import image as mp_image
print('mp_image', mp_image.Image)
