from picamera2 import Picamera2
from datetime import datetime
from picamera2.encoders import H264Encoder
import time
import os

def record_video():
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(
        main={"format": 'RGB888', "size": (1080, 1080)})
    picam2.set_controls({"AfMode": 0, "LensPosition": 0.0})
    picam2.configure(camera_config)
    picam2.start()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/home/bplusplus/Videos/sensing-garden/video_{timestamp}.mp4"
    encoder = H264Encoder(bitrate=10000000)  # 10 Mbps

    picam2.start_recording(encoder, filename)
    time.sleep(60)  # Record for 1 minute (60 seconds)
    picam2.stop_recording()
    picam2.close()

def main():
    if not os.path.exists("/home/bplusplus/Videos/sensing-garden"):
        os.makedirs("/home/bplusplus/Videos/sensing-garden")

    while True:
        record_video()
        time.sleep(540)  # Sleep for 9 minutes (540 seconds) 

if __name__ == "__main__":
    main()

