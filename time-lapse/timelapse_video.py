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
    print("Entering record_video", flush=True)
    picam2 = Picamera2()
    print("Picamera2 instance created", flush=True)
    camera_config = picam2.create_video_configuration(
        main={"format": 'RGB888', "size": (1080, 1080)})
    print("Video configuration created", flush=True)
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
    print("Camera controls set", flush=True)
    picam2.configure(camera_config)
    print("Camera configured", flush=True)
    picam2.start()
    print("Camera started", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(video_dir, f"video_{timestamp}.mp4")
    encoder = H264Encoder(bitrate=10000000)  # 10 Mbps
    print("Encoder created", flush=True)

    picam2.start_recording(encoder, filename)
    print(f"Started recording to {filename}", flush=True)
    print("Sleeping for 5 seconds while recording", flush=True)
    time.sleep(5)  # Record for 5 seconds
    print("Woke up from sleep, stopping recording", flush=True)
    picam2.stop_recording()
    print("Recording stopped", flush=True)
    picam2.close()
    print("Camera closed", flush=True)
    print(f"Video saved to {filename}", flush=True)
    print("Exiting record_video", flush=True)

from typing import Optional


def upload_video(
    video_path: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    print("Entering upload_video", flush=True)
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
    print("Upload response:", response, flush=True)
    print("Exiting upload_video", flush=True)


def get_unuploaded_videos(directory: str):
    print("Entering get_unuploaded_videos", flush=True)
    result = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".mp4")]
    print(f"Exiting get_unuploaded_videos with {len(result)} videos", flush=True)
    return result

def main():
    print("Entering main", flush=True)
    while True:
        current_hour = datetime.now().hour
        # if 6 <= current_hour < 22:  # Check if the current time is between 6:00 AM and 10:00 PM
        record_video()
        # After recording, upload all un-uploaded videos
        unuploaded = get_unuploaded_videos(video_dir)
        for video_path in sorted(unuploaded, key=os.path.getmtime):
            try:
                upload_video(video_path=video_path)
                # After successful upload, delete the video file
                try:
                    os.remove(video_path)
                    print(f"Deleted video: {video_path}", flush=True)
                except Exception as del_exc:
                    print(f"Failed to delete {video_path}: {del_exc}", flush=True)
            except Exception as e:
                print(f"Failed to upload {video_path}: {e}", flush=True)
        # else:
        #     print("Outside active hours. Waiting...")
        # time.sleep(540)  # Sleep for 9 minutes (540 seconds)

if __name__ == "__main__":
    main()

