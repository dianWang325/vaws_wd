import importlib.util,json,os,runpy
import errno,pathlib,tempfile
_cleanup=tempfile.TemporaryDirectory.cleanup
def nfs_cleanup(self):
    try:
        _cleanup(self)
    except OSError as exc:
        path=pathlib.Path(self.name)
        if exc.errno!=errno.ENOTEMPTY or path.parent.resolve()!=pathlib.Path('/tmp') or not path.is_dir():
            raise
        remaining=list(path.iterdir())
        if not remaining or not all(p.is_file() and p.name.startswith('.nfs') for p in remaining):
            raise
        print('NFS_TEMP_DEFERRED_CLEANUP',str(path),flush=True)
tempfile.TemporaryDirectory.cleanup=nfs_cleanup
from vllm.utils import network_utils as net
def tcp_store():
    return net.get_distributed_init_method('127.0.0.1',net.get_open_port())
net.get_file_store_init_method=tcp_store
from vllm.v1.executor import multiproc_executor
multiproc_executor.get_file_store_init_method=tcp_store
if __name__=='__main__':
    print('SOURCE_PROVENANCE',json.dumps({n:importlib.util.find_spec(n).origin for n in ['vllm','vllm_ascend']}),flush=True)
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
