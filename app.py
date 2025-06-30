from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
from datetime import datetime
import argparse
import uuid
import os
import shutil

CONFIG_PATH = Path("configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")

app = FastAPI()


def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
):
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )

    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed)

    try:
        main(config=config, args=args)
        print("Processing completed successfully.")
        return output_path
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise RuntimeError(f"Error during processing: {str(e)}")


def create_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, guidance_scale: float, seed: int
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            CHECKPOINT_PATH.absolute().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--temp_dir",
            "temp",
            "--enable_deepcache",
        ]
    )


@app.post("/process")
async def process(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    guidance_scale: float = Form(1.5),
    inference_steps: int = Form(20),
    seed: int = Form(1247),
):
    temp_dir = "temp_inputs"
    os.makedirs(temp_dir, exist_ok=True)

    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{video.filename}")
    audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{audio.filename}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    with open(audio_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    output_path = process_video(
        video_path=video_path,
        audio_path=audio_path,
        guidance_scale=guidance_scale,
        inference_steps=inference_steps,
        seed=seed,
    )

    return FileResponse(output_path, media_type="video/mp4", filename="output.mp4")
# uvicorn main:app --host 0.0.0.0 --port 8000