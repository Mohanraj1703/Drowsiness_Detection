import mediapipe as mp
print('version', mp.__version__)
print('has solutions', hasattr(mp, 'solutions'))
print('dir', [x for x in dir(mp) if 'face' in x.lower() or 'solution' in x.lower()])
