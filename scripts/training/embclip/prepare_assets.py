"""Public EmbCLIP assets in independent leaves; never run Unity on the login node."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / 'assets/embclip-robothor'
DATA = ROOT / 'data/embclip-robothor/robothor-objectnav'
EPISODES_URL = ('https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/'
                'navigation/robothor-objectnav-challenge-2021.tar.gz')
CHECKPOINT_URL = ('https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/'
                  'navigation/exp_Objectnav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO'
                  '__stage_00__steps_000130091717.pt')
TARGETS = {'AlarmClock', 'Apple', 'BaseballBat', 'BasketBall', 'Bowl', 'GarbageCan',
           'HousePlant', 'Laptop', 'Mug', 'SprayBottle', 'Television', 'Vase'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = destination.with_name(destination.name + '.json')
    if destination.exists():
        old = json.loads(receipt.read_text())
        assert old['url'] == url and old['sha256'] == digest(destination), destination
        assert old['bytes'] == destination.stat().st_size
        return old
    partial = destination.with_name(destination.name + '.part')
    # Preserve interrupted downloads for diagnosis rather than silently overwriting them.
    with urlopen(url, timeout=120) as response, partial.open('xb') as out:
        expected = response.headers.get('Content-Length')
        shutil.copyfileobj(response, out)
    assert partial.stat().st_size > 0
    if expected is not None:
        assert partial.stat().st_size == int(expected), 'incomplete download'
    info = dict(url=url, bytes=partial.stat().st_size, sha256=digest(partial),
                checksum_source='locally computed; upstream checksum not supplied')
    receipt.write_text(json.dumps(info, indent=2) + '\n')
    partial.rename(destination)
    return info


def stripped_member(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts, name
    return Path(*path.parts[1:]) if len(path.parts) > 1 else None


def extract_episodes(archive):
    if DATA.exists():
        assert (DATA / 'train/episodes').is_dir() and (DATA / 'val/episodes').is_dir()
        return
    temporary = DATA.with_name(DATA.name + '.partial')
    temporary.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, 'r:gz') as tar:
        roots = set()
        for member in tar:
            relative = stripped_member(member.name)
            roots.add(PurePosixPath(member.name).parts[0])
            assert len(roots) == 1, 'unexpected archive layout'
            assert member.isdir() or member.isfile(), 'links/devices are forbidden'
            if relative is None:
                continue
            output = temporary / relative
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, output.open('xb') as target:
                    shutil.copyfileobj(source, target)
    assert (temporary / 'train/episodes').is_dir()
    assert (temporary / 'val/episodes').is_dir()
    temporary.rename(DATA)


def verify_episodes():
    splits = {}
    for split in ('train', 'val'):
        files = sorted((DATA / split / 'episodes').glob('*.json.gz'))
        assert files, 'missing ' + split
        count, categories = 0, set()
        for path in files:
            with gzip.open(path, 'rt') as f:
                episodes = json.load(f)  # Reading to EOF also checks the gzip CRC.
            assert isinstance(episodes, list) and episodes, path
            ids = set()
            for episode in episodes:
                assert {'id', 'initial_position', 'initial_orientation', 'object_type'} <= episode.keys()
                assert episode['object_type'] in TARGETS, episode['object_type']
                if 'scene' in episode:
                    assert episode['scene'] == path.name[:-8], (path, episode['scene'])
                assert episode['id'] not in ids, 'duplicate episode ID in ' + str(path)
                ids.add(episode['id'])
                categories.add(episode['object_type'])
            count += len(episodes)
        assert categories == TARGETS, (split, categories)
        splits[split] = dict(scenes=[p.name[:-8] for p in files], episodes=count,
                             categories=sorted(categories))
    assert set(splits['train']['scenes']).isdisjoint(splits['val']['scenes'])
    files = sorted(p for p in DATA.rglob('*') if p.is_file())
    return dict(splits=splits, file_count=len(files), bytes=sum(p.stat().st_size for p in files),
                files={str(p.relative_to(DATA)): digest(p) for p in files})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['episodes', 'checkpoint', 'clip', 'verify'])
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    if args.stage == 'episodes':
        archive = ASSETS / 'downloads/robothor-objectnav-challenge-2021.tar.gz'
        download(EPISODES_URL, archive)
        extract_episodes(archive)
    if args.stage in ('episodes', 'verify'):
        report = verify_episodes()
    elif args.stage == 'checkpoint':
        report = download(CHECKPOINT_URL, ASSETS / 'checkpoints/clip-rn50-gru-130091717.pt')
    else:
        import clip
        url = clip.clip._MODELS['RN50']
        path = ASSETS / 'clip/RN50.pt'
        report = download(url, path)
        assert report['sha256'] == url.split('/')[-2], 'OpenAI RN50 checksum mismatch'
        report['checksum_source'] = 'SHA256 embedded in official OpenAI CLIP model URL'
    (ASSETS / ('verified-' + args.stage + '.json')).write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report if args.stage not in ('episodes', 'verify') else
                     {k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
