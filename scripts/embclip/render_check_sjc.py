"""Render a real RoboTHOR scene on the SJC GPU workstation."""
import argparse
import ctypes.util
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['cloud', 'legacy-headless'], default='cloud')
    parser.add_argument('--scene', default='FloorPlan_Val1_1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    cache = root / 'assets/embclip-robothor/ai2thor'
    cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {'status': 'failed', 'mode': args.mode, 'scene': args.scene,
              'execution_host': 'sjc-remote', 'node': socket.gethostname(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'vulkan_loader': ctypes.util.find_library('vulkan')}
    controller = None
    try:
        import ai2thor
        import ai2thor.build
        from ai2thor.controller import Controller
        from ai2thor.platform import CloudRendering
        from PIL import Image
        import numpy as np

        Controller.base_dir = property(lambda self: str(cache))
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                              '--format=csv,noheader'], capture_output=True, text=True, check=True)
        result.update(ai2thor_version=ai2thor.__version__, gpu=gpu.stdout.strip())
        kwargs = dict(scene=args.scene, width=400, height=300, agentMode='locobot',
                      gridSize=0.25, rotateStepDegrees=30, visibilityDistance=1.0,
                      continuousMode=True, snapToGrid=False, applyActionNoise=True,
                      fieldOfView=math.degrees(2 * math.atan(math.tan(math.radians(79) / 2) * 300 / 400)), renderDepthImage=False,
                      include_private_scenes=False)
        if args.mode == 'cloud':
            kwargs.update(platform=CloudRendering, commit_id=ai2thor.build.COMMIT_ID)
        else:
            kwargs.update(platform=CloudRendering, commit_id='91139c909576f3bf95a187c5b02c6fd455d06b48')
        result['simulator_commit'] = kwargs['commit_id']
        controller = Controller(**kwargs)
        initial = controller.last_event.frame
        event = controller.step(action='RotateRight', degrees=30)
        frame = event.frame
        assert frame.shape == (300, 400, 3), frame.shape
        assert frame.dtype == np.uint8 and float(frame.std()) > 1, 'invalid/blank RGB frame'
        assert event.metadata['lastActionSuccess'], event.metadata.get('errorMessage')
        assert len(event.metadata.get('objects', [])) > 0, 'RoboTHOR scene metadata absent'
        assert np.any(initial != frame), 'action produced no image change'
        image = args.output / 'rgb.png'
        Image.fromarray(frame).save(image)
        result.update(status='complete', completed_checks=1, action='RotateRight',
                      action_success=True, rgb_shape=list(frame.shape),
                      rgb_std=float(frame.std()), rgb_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                      policy_inputs=['rgb', 'target_category'], policy_executed=False)
    except BaseException:
        result['error'] = traceback.format_exc()
        (args.output / 'error.txt').write_text(result['error'])
        raise
    finally:
        if controller is not None:
            controller.stop()
        result['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
