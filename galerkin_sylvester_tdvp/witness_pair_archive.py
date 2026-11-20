"""Pack finalized A/B witness worker directories into verified family/size ZIPs.

From the repository root:
  python3 -m galerkin_sylvester_tdvp.witness_pair_archive pack RUN
  python3 -m galerkin_sylvester_tdvp.witness_pair_archive verify RUN

Each ZIP member name is relative to RUN. Failures and accepted checkpoints
are preserved. Archiving leaves the original worker files on disk.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.+=-]*\Z')


def digest(stream):
    h = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            return h.hexdigest()
        h.update(chunk)


def file_digest(path):
    with path.open('rb') as f:
        return digest(f)


def collect(run):
    """Read the final enrollment and validate each worker's terminal record."""
    index = run / 'measurements.json'
    grid = json.loads((run / 'grid.json').read_text())
    expected = [(c['phase'], c['config_id'], i) for c in grid
                for i in range(1, c['requested_repeats'] + 1)]
    rows = json.loads(index.read_text())
    observed = [(r['phase'], r['config_id'], r['repeat']) for r in rows]
    if not rows or len(expected) != len(set(expected)) or sorted(expected) != sorted(observed):
        raise ValueError('Enrollment and terminal repeat IDs differ')
    grouped = defaultdict(list)
    names = set()
    for row in rows:
        name = row['directory']
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or name in names:
            raise ValueError(f'Invalid or repeated worker directory {name!r}')
        names.add(name)
        family, n = row['family'], row['n']
        if family not in ('D', 'E') or type(n) is not int or n < 1:
            raise ValueError(f'Invalid family/size in {name}')
        folder = run / name
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f'Missing or linked worker directory {name}')
        if row.get('parent_complete') is not True or json.loads((folder / 'measurement.json').read_text()) != row:
            raise ValueError(f'Worker and final aggregate disagree for {name}')
        grouped[f'{family}{n:02d}'].append(name)
    return grouped


def inventory(run, names):
    entries = {}
    for name in sorted(names):
        for p in sorted((run / name).rglob('*')):
            if p.is_symlink() or not (p.is_file() or p.is_dir()):
                raise ValueError(f'Unexpected nonregular worker entry {p}')
            if p.is_file():
                entries[p.relative_to(run).as_posix()] = file_digest(p)
    return entries


def verify(raw):
    """Verify every archived path/content hash, also on a fresh checkout."""
    manifest = json.loads((raw / 'manifest.json').read_text())
    if manifest.get('version') != 1 or not manifest.get('archives'):
        raise ValueError('Unknown or empty archive manifest')
    if set(manifest.get('index_sha256', {})) != {'grid.json', 'measurements.json'}:
        raise ValueError('Unexpected index checksum list')
    sets = manifest['archives']
    expected = ''.join(f"{entry['sha256']}  {name}\n" for name, entry in sorted(sets.items()))
    if (raw / 'SHA256SUMS').read_text() != expected:
        raise ValueError('SHA256SUMS and manifest disagree')
    if {p.name for p in raw.glob('*.zip')} != set(sets):
        raise ValueError('Unexpected or missing ZIP')
    all_workers = set()
    for name, entry in sets.items():
        if not re.fullmatch(r'[DE][0-9]+\.zip', name) or not entry['workers']:
            raise ValueError('Invalid archive entry')
        path = raw / name
        if file_digest(path) != entry['sha256']:
            raise ValueError(f'ZIP checksum mismatch for {name}')
        workers = set(entry['workers'])
        if (len(workers) != len(entry['workers']) or all_workers & workers or
                any(not SAFE_NAME.fullmatch(w) for w in workers)):
            raise ValueError(f'Invalid worker list in {name}')
        all_workers.update(workers)
        with ZipFile(path) as z:
            members = z.namelist()
            if len(set(members)) != len(members) or set(members) != set(entry['files']):
                raise ValueError(f'ZIP membership mismatch in {name}')
            for member in members:
                parts = member.split('/')
                if len(parts) < 2 or parts[0] not in workers or any(k in ('', '.', '..') for k in parts):
                    raise ValueError(f'Invalid member path {member}')
                with z.open(member) as stream:
                    if digest(stream) != entry['files'][member]:
                        raise ValueError(f'File checksum mismatch for {member}')
            if any(w + '/measurement.json' not in entry['files'] for w in workers):
                raise ValueError(f'Missing terminal record in {name}')
    return manifest


def ignore_verified(run, names):
    lines = [('/' + name + '/').encode('ascii') for name in sorted(names)]
    target = run / '.gitignore'
    if target.is_symlink():
        raise ValueError('Refusing to modify linked .gitignore')
    previous = target.read_bytes() if target.exists() else b''
    missing = [line for line in lines if line not in previous.splitlines()]
    if missing:
        with target.open('ab') as f:
            if previous and not previous.endswith(b'\n'):
                f.write(b'\n')
            f.write(b'# Loose worker directories preserved byte-for-byte in raw/*.zip\n')
            f.write(b'\n'.join(missing) + b'\n')


def pack(run):
    """Publish a validated archive set atomically; refuse conflicting existing archives."""
    control = ('grid.json','measurements.json')
    baseline = {f: file_digest(run / f) for f in control}
    groups = collect(run)
    raw = run / 'raw'
    if raw.is_symlink():
        raise ValueError('Refusing to follow linked archive directory')
    files = {name + '.zip': inventory(run, workers) for name, workers in groups.items()}
    if not raw.exists():
        with TemporaryDirectory(prefix='.pack-witness-', dir=run) as temp:
            staged = Path(temp) / 'raw'
            staged.mkdir()
            sets = {}
            for name, members in sorted(files.items()):
                archive = staged / name
                with ZipFile(archive, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
                    for member in members:
                        z.write(run / member, member)
                sets[name] = dict(sha256=file_digest(archive), files=members,
                                  workers=sorted(groups[name[:-4]]))
            manifest = dict(version=1, index_sha256=baseline, archives=sets)
            (staged / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
            (staged / 'SHA256SUMS').write_text(''.join(
                f"{e['sha256']}  {name}\n" for name, e in sets.items()))
            example = next(iter(sets))
            (staged / 'README.md').write_text(
                '# Raw witness workers\n\nEach ZIP groups all enrolled A and B processes for one '
                'physical family and spin count. Member paths preserve the original process '
                'directories. The manifest records SHA-256 checksums for each ZIP, each member '
                'and the two top-level measurement indexes. Timeout, resource and numerical '
                'failures retain their original records.\n\nFrom the repository root, set '
                '`RUN` to the directory containing this `raw` folder. '
                'Verify the archives, then expand a family/size into a fresh destination.\n\n'
                '```sh\npython3 -m galerkin_sylvester_tdvp.witness_pair_archive verify "$RUN"\n'
                f'unzip "$RUN/raw/{example}" -d /tmp/new-witness-workers\n```\n')
            verify(staged)
            if any(inventory(run, groups[name[:-4]]) != entries for name, entries in files.items()):
                raise ValueError('Loose worker files changed while packing')
            if baseline != {f: file_digest(run / f) for f in control}:
                raise ValueError('Run indexes changed while packing')
            staged.rename(raw)
    published = verify(raw)
    if published['index_sha256'] != baseline or set(files) != set(published['archives']):
        raise ValueError('Existing archives belong to a different run index')
    for name, original in files.items():
        if published['archives'][name]['files'] != original or inventory(run, groups[name[:-4]]) != original:
            raise ValueError(f'Archive differs from current loose files in {name}')
    ignore_verified(run, (w for group in groups.values() for w in group))
    return published


def check_run(run):
    manifest = verify(run / 'raw')
    if any(file_digest(run / f) != d for f, d in manifest['index_sha256'].items()):
        raise ValueError('Measurement or enrollment index checksum mismatch')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('pack','verify'))
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    manifest = (pack if args.action == 'pack' else check_run)(args.run.resolve())
    sets = manifest['archives']
    print(f"{len(sets)} verified ZIPs, {sum(len(r['workers']) for r in sets.values())} worker directories")
