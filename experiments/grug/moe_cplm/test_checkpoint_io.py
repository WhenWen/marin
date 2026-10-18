# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU storage/reference checks only: no optimizer updates or training loop."""
import hashlib
import json
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.checkpoint import save_checkpoint,load_checkpoint
from levanter.tensorstore_serialization import TensorStoreWriteConfig
from experiments.grug.moe_cplm.checkpoint_io import python_zarr_checkpoint_writes


def local_state_digest(state):
    digest = hashlib.sha256()
    for path, value in jax.tree_util.tree_leaves_with_path(state):
        array = np.asarray(value)
        digest.update(str((path, array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def test_native_readback_repeated_full_state(tmp_path):
    state={'step':jnp.array(1650),'params':{'w':jnp.arange(65536,dtype=jnp.float32).reshape(256,256),
        'b':jnp.arange(63,dtype=jnp.bfloat16).reshape(7,9)},'opt':{'mom':jnp.ones((32,64)),
        'rng':jax.random.PRNGKey(17),'count':jnp.array(1650),'zero':jnp.zeros((64,32))}}
    expected=local_state_digest(state)
    with python_zarr_checkpoint_writes():
        for n in range(3):
            path=tmp_path/f'step-{n}'
            save_checkpoint(state,n,str(path),is_temporary=False,write_config=TensorStoreWriteConfig(max_chunk_bytes=1024,max_staged_host_bytes=4096))
            assert not (path/'manifest.ocdbt').exists()
            assert json.loads((path/'metadata.json').read_text())['step']==n
            assert local_state_digest(load_checkpoint(state,str(path),allow_partial=False))==expected


def test_upload_failure_never_publishes_complete_marker(tmp_path):
    from fsspec.implementations.local import LocalFileSystem
    with python_zarr_checkpoint_writes(),patch.object(LocalFileSystem,'pipe_file',side_effect=OSError('injected upload failure')):
        with pytest.raises(OSError,match='injected upload failure'):
            save_checkpoint({'x':jnp.arange(16)},0,str(tmp_path/'failed'),is_temporary=False)
    assert not (tmp_path/'failed/metadata.json').exists()


def test_distributed_chunk_ownership_on_four_cpu_devices(tmp_path):
    if len(jax.devices()) < 4:
        pytest.skip('Requires four virtual CPU devices for shard ownership readback')
    from jax.sharding import NamedSharding,Mesh,PartitionSpec as P
    mesh=Mesh(np.array(jax.devices()[:4]),('s',));sharding=NamedSharding(mesh,P('s',None))
    x=jax.device_put(np.arange(256*128,dtype=np.float32).reshape(256,128),sharding)
    with python_zarr_checkpoint_writes():
        save_checkpoint({'x':x},0,str(tmp_path/'sharded'),is_temporary=False,write_config=TensorStoreWriteConfig(max_chunk_bytes=1024,max_staged_host_bytes=4096))
    restored=load_checkpoint({'x':x},str(tmp_path/'sharded'),allow_partial=False)
    np.testing.assert_array_equal(np.asarray(restored['x']),np.asarray(x))
