import os
import time
from datetime import datetime

import sensing_garden_client as sgc
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder

# Define video_dir at the top level, relative to this script
video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "videos"))
os.makedirs(video_dir, exist_ok=True)

def record_video():
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(
        main={"format": 'RGB888', "size": (1080, 1080)})
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
    picam2.configure(camera_config)
    picam2.start()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(video_dir, f"video_{timestamp}.mp4")
    encoder = H264Encoder(bitrate=10000000)  # 10 Mbps

    picam2.start_recording(encoder, filename)
    time.sleep(60)  # Record for 1 minute (60 seconds)
    picam2.stop_recording()
    picam2.close()

from typing import Optional


def upload_video(
    video_path: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Uploads a video file to the Sensing Garden backend.
    Loads config from environment variables. If video_path is None, uploads the latest video in the default directory.
    """
    import os
    from datetime import datetime

    from sensing_garden_client import SensingGardenClient

    # Load config from environment variables
    api_key = os.environ.get("SENSING_GARDEN_API_KEY")
    base_url = os.environ.get("API_BASE_URL")
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    device_id = os.environ.get("DEVICE_ID")

    if not all([api_key, base_url, aws_access_key_id, aws_secret_access_key, device_id]):
        raise RuntimeError("Missing one or more required environment variables for video upload.")
    
    # Determine video file to upload
    if video_path is None:
        mp4_files = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith(".mp4")
        ]
        if not mp4_files:
            raise FileNotFoundError(f"No .mp4 files found in {video_dir}")
        video_path = max(mp4_files, key=os.path.getmtime)

    # Read video data
    with open(video_path, "rb") as f:
        video_data = f.read()

    # Set up client
    sgc = SensingGardenClient(
        base_url=base_url,
        api_key=api_key,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )

    timestamp = datetime.now().isoformat()
    upload_metadata = metadata or {}
    response = sgc.videos.upload_video(
        device_id=device_id,
        timestamp=timestamp,
        video_path_or_data=video_data,
        content_type="video/mp4",
        metadata=upload_metadata
    )
    print("Upload response:", response)


def get_unuploaded_videos(directory: str):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".mp4")]

def main():
    while True:
        current_hour = datetime.now().hour
        if 6 <= current_hour < 22:  # Check if the current time is between 6:00 AM and 10:00 PM
            record_video()
            # After recording, upload all un-uploaded videos
            unuploaded = get_unuploaded_videos(video_dir)
            for video_path in sorted(unuploaded, key=os.path.getmtime):
                try:
                    upload_video(video_path=video_path)
                    # After successful upload, delete the video file
                    try:
                        os.remove(video_path)
                        print(f"Deleted video: {video_path}")
                    except Exception as del_exc:
                        print(f"Failed to delete {video_path}: {del_exc}")
                except Exception as e:
                    print(f"Failed to upload {video_path}: {e}")
        else:
            print("Outside active hours. Waiting...")
        time.sleep(540)  # Sleep for 9 minutes (540 seconds)

if __name__ == "__main__":
    main()

