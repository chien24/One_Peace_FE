import subprocess

subprocess.run([
    "ffmpeg",
    "-ss", "30",
    "-i", r"samples\test_video\_gQFB_Utuf0_30.000_40.000.mp4",
    "-t", "10",
    "-c", "copy",
    "output.mp4"
])