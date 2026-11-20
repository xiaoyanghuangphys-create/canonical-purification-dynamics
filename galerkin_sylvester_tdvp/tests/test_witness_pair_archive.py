"""Integrity, immutability and replay of the family/size raw archives."""
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from galerkin_sylvester_tdvp.witness_pair_archive import check_run, pack


def fixture(root):
    cfg = [dict(phase='screen', config_id=name, requested_repeats=1)
           for name in ('D8_A', 'D8_B', 'E16_A')]
    rows = []
    for c in cfg:
        ident = c['config_id']
        name = f'screen_{ident}_r1'
        folder = root / name
        folder.mkdir()
        row = dict(c, family=ident[0], n=int(ident[1:].split('_')[0]),
                   repeat=1, parent_complete=True, directory=name,
                   status='timeout' if ident.endswith('_B') else 'completed')
        (folder / 'measurement.json').write_text(json.dumps(row))
        (folder / 'events.jsonl').write_bytes(b'\x00\n')
        (folder / 'nested').mkdir()
        (folder / 'nested' / 'trace.log').write_bytes(b'raw\xff')
        rows.append(row)
    (root / 'grid.json').write_text(json.dumps(cfg))
    (root / 'measurements.json').write_text(json.dumps(rows))
    return rows


class WitnessArchiveTests(unittest.TestCase):
    def test_roundtrip_and_idempotent_preservation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = fixture(root)
            (root / 'surprise').mkdir()
            (root / '.gitignore').write_bytes(b'# prior content')
            manifest = pack(root)
            ignores = (root / '.gitignore').read_bytes()
            self.assertEqual(set(manifest['archives']), {'D08.zip','E16.zip'})
            self.assertTrue(ignores.startswith(b'# prior content\n'))
            self.assertNotIn(b'surprise', ignores)
            self.assertEqual(manifest, pack(root))
            self.assertEqual(ignores, (root / '.gitignore').read_bytes())
            for row in rows:
                self.assertIn('/' + row['directory'] + '/', ignores.decode().splitlines())
                self.assertTrue((root / row['directory']).is_dir())
                shutil.rmtree(root / row['directory'])
            self.assertEqual(check_run(root),manifest)
            dest = root / 'replay'
            with ZipFile(root / 'raw/D08.zip') as z:
                z.extractall(dest)
            rescued = 'screen_D8_B_r1'
            self.assertEqual((dest / rescued / 'nested/trace.log').read_bytes(),b'raw\xff')
            self.assertEqual(json.loads((dest / rescued / 'measurement.json').read_text())['status'],'timeout')

    def test_incomplete_or_changed_worker_fails(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=fixture(root)
            (root / 'measurements.json').write_text(json.dumps(rows[:-1]))
            with self.assertRaisesRegex(ValueError,'Enrollment'):
                pack(root)
            self.assertFalse((root / 'raw').exists())
            self.assertFalse((root / '.gitignore').exists())
            (root / 'measurements.json').write_text(json.dumps(rows))
            pack(root)
            archive=root / 'raw/D08.zip'
            before=hashlib.sha256(archive.read_bytes()).digest()
            (root / rows[0]['directory'] / 'events.jsonl').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'current loose files'):
                pack(root)
            self.assertEqual(before,hashlib.sha256(archive.read_bytes()).digest())
            archive.write_bytes(b'damaged')
            with self.assertRaisesRegex(ValueError,'checksum'):
                check_run(root)

    def test_filesystem_and_record_disagreements(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = fixture(root)
            row=rows[0]
            p=root / row['directory'] / 'measurement.json'
            p.write_text(json.dumps(dict(row, status='failed')))
            with self.assertRaisesRegex(ValueError,'disagree'):
                pack(root)
            p.write_text(json.dumps(row))
            (p.parent/'unrelated').symlink_to(root/'grid.json')
            with self.assertRaisesRegex(ValueError,'nonregular'):
                pack(root)
            self.assertFalse((root/'.gitignore').exists())
