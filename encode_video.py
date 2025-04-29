import base64

with open("heart-animation.mp4", "rb") as f:
    video_bytes = f.read()
    encoded = base64.b64encode(video_bytes).decode()

with open("video_base64.txt", "w") as out:
    out.write(encoded)
