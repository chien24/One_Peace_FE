import yt_dlp
import os

id = '_gQFB_Utuf0_30.000_40.000'
url = f"https://www.youtube.com/watch?v={id}"

ydl_opts = {
    "format": "bestvideo+bestaudio/best",
    "outtmpl": f"{id}.%(ext)s",
    "merge_output_format": "mp4",
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])