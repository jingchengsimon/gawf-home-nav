"""Create an isolated official RoboTHOR Challenge 2021 client/build on SJC.

The existing training environment supplies read-only shared dependencies. The
new virtualenv owns its pinned ai2thor package and an independent asset cache.
No package in the source environment is installed, removed or upgraded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen
import zipfile

BUILD = 'bad5bc2b250615cb766ffb45d455c211329af17e'
ZIP_SHA = 'fba7b5f2fbb7a48f37d39cbf17d0d9301092945aaa0dd7215b20a85a421c287c'
ZIP_BYTES = 579427830
WHEEL_SHA = '0553de4c1c43da896f9aea9d440254235689cafcd6162a723735c79662c865b0'
WHEEL_URL = 'https://files.pythonhosted.org/packages/0c/e1/c710464959424f13e18bbcb0ddf4fb2459c4966a44e456083f9e10d39619/ai2thor-2.7.2-py2.py3-none-any.whl'
ZIP_URL = 'https://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/thor-Linux64-' + BUILD + '.zip'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def download(url, path, digest, size=None):
    if path.exists():
        if sha(path) != digest:
            raise RuntimeError('Existing asset has wrong hash: ' + str(path))
        return
    part = path.with_suffix(path.suffix + '.incoming')
    if part.exists():
        raise RuntimeError('Retained partial download requires an explicit resume: ' + str(part))
    with urlopen(url, timeout=60) as response, part.open('xb') as stream:
        while True:
            block = response.read(4 * 1024 * 1024)
            if not block:
                break
            stream.write(block)
    if (size is not None and part.stat().st_size != size) or sha(part) != digest:
        raise RuntimeError('Downloaded asset failed size/hash check: ' + str(part))
    part.rename(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out.parent != root / 'results' or out.is_symlink():
        raise RuntimeError('Output must be an independent result leaf')
    out.mkdir(exist_ok=True)
    if (out / 'summary.json').exists():
        raise RuntimeError('Output already finalized')
    source_python = root / '.envs/embclip-robothor/bin/python'
    target = root / '.envs/embclip-robothor-challenge2021'
    assets = root / 'assets/embclip-robothor-challenge2021'
    before = subprocess.check_output([str(source_python), '-m', 'pip', 'freeze'])
    (out / 'training-runtime-before.txt').write_bytes(before)
    if target.exists() or assets.exists():
        raise RuntimeError('Independent runtime/assets already exist; refusing to overwrite')
    if shutil.disk_usage(root).free < ZIP_BYTES * 6:
        raise RuntimeError('Insufficient disk for archive and extraction')
    assets.mkdir()
    cache = assets / 'downloads'
    cache.mkdir()
    try:
        wheel = cache / 'ai2thor-2.7.2-py2.py3-none-any.whl'
        archive = cache / ('thor-Linux64-' + BUILD + '.zip')
        print('Downloading hash-pinned official client/build', flush=True)
        download(WHEEL_URL, wheel, WHEEL_SHA, 51260)
        download(ZIP_URL, archive, ZIP_SHA, ZIP_BYTES)
        incoming = assets / 'build.incoming'
        incoming.mkdir()
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                dest = (incoming / member.filename).resolve()
                if incoming.resolve() not in dest.parents and dest != incoming.resolve():
                    raise RuntimeError('Archive member escapes destination')
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError('Unexpected symlink in simulator archive')
            z.extractall(incoming)
            for member in z.infolist():
                dest = incoming / member.filename
                mode = (member.external_attr >> 16) & 0o777
                if dest.is_file() and mode:
                    dest.chmod(mode)
        executable = next(x for x in incoming.rglob('thor-Linux64-' + BUILD) if x.is_file())
        executable.chmod(0o755)
        rel_executable = executable.relative_to(incoming)
        build = assets / 'build'
        incoming.rename(build)
        executable = build / rel_executable
        subprocess.run([str(source_python), '-m', 'venv', str(target)], check=True)
        python = target / 'bin/python'
        site = next((target / 'lib').glob('python*/site-packages'))
        base_site = subprocess.check_output([str(source_python), '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True).strip()
        (site / 'shared_readonly_dependencies.pth').write_text(base_site + '\n')
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps', '--ignore-installed', str(wheel)], check=True)
        info = json.loads(subprocess.check_output([str(python), '-c', 'import ai2thor,json,torch;print(json.dumps(dict(ai2thor=ai2thor.__version__,ai2thor_file=ai2thor.__file__,torch=torch.__version__)))'], text=True))
        if info['ai2thor'] != '2.7.2' or target not in Path(info['ai2thor_file']).parents:
            raise RuntimeError('Independent client import failed')
        after = subprocess.check_output([str(source_python), '-m', 'pip', 'freeze'])
        (out / 'training-runtime-after.txt').write_bytes(after)
        if after != before:
            raise RuntimeError('Training environment changed')
        summary = dict(status='complete', runtime=str(target.relative_to(root)),
            python=str(python.relative_to(root)), executable=str(executable.relative_to(root)),
            executable_sha256=sha(executable), build_commit=BUILD,
            archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
            ai2thor_version=info['ai2thor'], torch_version=info['torch'],
            training_runtime_unchanged=True, weights_updated=False,
            shared_dependencies='Existing training site-packages are read-only, after new runtime site-packages on sys.path.')
        write(out / 'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write(out / 'failure.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    main()
