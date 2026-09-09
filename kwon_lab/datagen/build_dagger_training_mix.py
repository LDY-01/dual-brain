"""6cm 정상 시연과 DAgger 교정을 프레임 수 기준으로 균형 혼합한다."""

import argparse
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_ID = "kwonlab/so101_sim_pick_dagger_mix_v1"
TASK = "Pick up the red block and place it on the green target zone."
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
FEATURES = {
    "observation.images.wrist": {
        "dtype": "video", "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32", "shape": (6,), "names": JOINT_NAMES,
    },
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
}


def tensor_image_to_uint8(image):
    array = image.detach().cpu().numpy()
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return np.clip(np.rint(array * 255), 0, 255).astype(np.uint8)


def append_episode(source, source_episode, destination):
    indices = np.flatnonzero(
        np.asarray(source.hf_dataset["episode_index"]) == source_episode
    )
    for index in indices:
        sample = source[int(index)]
        destination.add_frame({
            "observation.images.wrist": tensor_image_to_uint8(
                sample["observation.images.wrist"]
            ),
            "observation.state": np.asarray(
                sample["observation.state"], dtype=np.float32
            ),
            "action": np.asarray(sample["action"], dtype=np.float32),
            "task": TASK,
        })
    destination.save_episode()
    return len(indices)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-root", type=Path, required=True)
    ap.add_argument("--dagger-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--normal-episodes", type=int, default=100)
    ap.add_argument("--dagger-episodes", type=int)
    ap.add_argument(
        "--batch-encoding-size", type=int, default=10,
        help="Encode this many episodes per video batch to avoid one encoder startup per episode.",
    )
    args = ap.parse_args()

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite mixed dataset: {output}")
    normal = LeRobotDataset(
        repo_id="kwonlab/so101_sim_pick_v24", root=args.normal_root.resolve(),
        video_backend="pyav",
    )
    dagger = LeRobotDataset(
        repo_id="kwonlab/so101_sim_pick_dagger_v1",
        root=args.dagger_root.resolve(), video_backend="pyav",
    )
    if args.normal_episodes > normal.num_episodes:
        raise ValueError("normal episode request exceeds dataset")

    # 원본 v2.4가 5x5x8 층화 후 셔플된 데이터이므로 전 구간에 걸쳐
    # 균등 간격으로 100개를 골라 원래의 위치·yaw 다양성을 유지한다.
    normal_ids = np.linspace(
        0, normal.num_episodes - 1, args.normal_episodes, dtype=int
    ).tolist()
    dagger_count = args.dagger_episodes or dagger.num_episodes
    if dagger_count > dagger.num_episodes:
        raise ValueError("dagger episode request exceeds dataset")
    dagger_ids = np.linspace(
        0, dagger.num_episodes - 1, dagger_count, dtype=int
    ).tolist()
    mixed = LeRobotDataset.create(
        REPO_ID, fps=25, features=FEATURES, root=output,
        robot_type="so101_sim",
        batch_encoding_size=args.batch_encoding_size,
    )

    normal_frames = dagger_frames = 0
    for number, episode in enumerate(normal_ids, 1):
        normal_frames += append_episode(normal, episode, mixed)
        if number % 10 == 0:
            print(f"normal {number}/{len(normal_ids)}")
    for number, episode in enumerate(dagger_ids, 1):
        dagger_frames += append_episode(dagger, episode, mixed)
        if number % 10 == 0:
            print(f"dagger {number}/{len(dagger_ids)}")

    total = normal_frames + dagger_frames
    manifest = {
        "format": "normal_6cm_plus_dagger_corrections",
        "normal_source": str(args.normal_root.resolve()),
        "dagger_source": str(args.dagger_root.resolve()),
        "normal_episode_ids": normal_ids,
        "dagger_episode_ids": dagger_ids,
        "normal_episodes": len(normal_ids),
        "dagger_episodes": len(dagger_ids),
        "normal_frames": normal_frames,
        "dagger_frames": dagger_frames,
        "total_episodes": len(normal_ids) + len(dagger_ids),
        "total_frames": total,
        "normal_frame_fraction": normal_frames / total,
        "dagger_frame_fraction": dagger_frames / total,
        "contains_12cm_data": False,
        "batch_encoding_size": args.batch_encoding_size,
    }
    (output / "mix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
