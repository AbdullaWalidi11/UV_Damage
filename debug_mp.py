import mediapipe
print(f"MediaPipe Version: {getattr(mediapipe, '__version__', 'Unknown')}")
print(f"Has 'solutions': {hasattr(mediapipe, 'solutions')}")
print(f"Has 'tasks': {hasattr(mediapipe, 'tasks')}")
print(f"Dir(mediapipe): {dir(mediapipe)}")

try:
    import mediapipe.python.solutions
    print("Success: import mediapipe.python.solutions")
except ImportError as e:
    print(f"Failed: import mediapipe.python.solutions -> {e}")

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("Success: import mediapipe.tasks.python.vision")
except ImportError as e:
    print(f"Failed: import mediapipe.tasks.python.vision -> {e}")
