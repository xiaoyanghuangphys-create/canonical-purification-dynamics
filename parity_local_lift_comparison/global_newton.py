"""Load only the two verified Newton files from their immutable Git objects.

The earlier research branch is deliberately separate. A shallow/filtered clone
without this object must fetch the named revision before running Newton rows.
"""
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType

REVISION='ba0d08c6b6cce07733204a7e2a9334dabb046134'
HASHES={'operator':'2399fa065ee42562332ac5be1e4f0106f6bac8d109f3704c3d75f5e3a299ec4a',
        'endpoint':'c154c0ecc672ed71bde3781e3fa7bedf6a392526da7b3bc87730ab7b64b2fc2b'}


def load_endpoint():
    prefix='_parity_comparison_verified_newton_'+REVISION
    if prefix+'.endpoint' in sys.modules:
        return sys.modules[prefix+'.endpoint'].run_endpoint
    root=Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as tmp:
        package=ModuleType(prefix)
        package.__path__=[tmp]
        package.__package__=prefix
        sys.modules[prefix]=package
        try:
            for name,expected in HASHES.items():
                blob=subprocess.check_output(['git','show',REVISION+':tangent_newton_witness/'+name+'.py'],cwd=root)
                if hashlib.sha256(blob).hexdigest()!=expected:
                    raise ValueError('Newton source failed SHA256 verification')
                path=Path(tmp)/(name+'.py')
                path.write_bytes(blob)
                spec=spec_from_file_location(prefix+'.'+name,path)
                module=module_from_spec(spec)
                sys.modules[spec.name]=module
                spec.loader.exec_module(module)
        except Exception:
            for key in [prefix,prefix+'.operator',prefix+'.endpoint']:
                sys.modules.pop(key,None)
            raise
    return sys.modules[prefix+'.endpoint'].run_endpoint
