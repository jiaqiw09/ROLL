"""
ref: https://github.com/volcengine/verl/blob/main/verl/protocol.py
Implement base data transfer protocol between any two functions, modules.
We can subclass Protocol to define more detailed batch info with specific keys
"""

import copy
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Set

import numpy as np
import ray
import tensordict
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from roll.utils.functionals import union_two_dict, divide_by_chunk_size
from roll.platforms import current_platform
from roll.utils.logging import get_logger

logger = get_logger()

try:
    tensordict.set_lazy_legacy(False).set()
except:
    pass


def _maybe_import_tq_types():
    try:
        from transfer_queue import BatchMeta, KVBatchMeta
        return BatchMeta, KVBatchMeta
    except Exception:
        return None, None


def dataflow_debug_enabled() -> bool:
    return os.getenv("ROLL_DEBUG_DATA_FLOW", "1").lower() not in {"0", "false", "off", "no"}


def _short_list(values, limit: int = 8) -> str:
    values = list(values)
    if len(values) <= limit:
        return str(values)
    return f"{values[:limit]}...(+{len(values) - limit})"


def describe_dataflow(data) -> str:
    BatchMeta, KVBatchMeta = _maybe_import_tq_types()

    if data is None:
        return "None"

    if isinstance(data, ray.ObjectRef):
        return "ObjectRef"

    if isinstance(data, (list, tuple)):
        sample_types = [type(item).__name__ for item in data[:5]]
        suffix = "" if len(data) <= 5 else f", +{len(data) - 5} more"
        return f"{type(data).__name__}(len={len(data)}, sample_types={sample_types}{suffix})"

    if isinstance(data, TensorDict):
        return f"TensorDict(batch_size={tuple(data.batch_size)}, keys={_short_list(data.keys())})"

    if BatchMeta is not None and isinstance(data, BatchMeta):
        field_names = list(getattr(data, "field_names", []) or [])
        extra_info = getattr(data, "extra_info", None) or {}
        return (
            f"BatchMeta(size={getattr(data, 'size', '?')}, "
            f"fields={_short_list(field_names)}, "
            f"extra_keys={_short_list(extra_info.keys())})"
        )

    if KVBatchMeta is not None and isinstance(data, KVBatchMeta):
        fields = list(getattr(data, "fields", []) or [])
        extra_info = getattr(data, "extra_info", None) or {}
        return (
            f"KVBatchMeta(keys={len(getattr(data, 'keys', []) or [])}, "
            f"partition_id={getattr(data, 'partition_id', None)}, "
            f"fields={_short_list(fields)}, "
            f"extra_keys={_short_list(extra_info.keys())})"
        )

    if isinstance(data, DataProto):
        raw_dict = object.__getattribute__(data, "__dict__")
        batch = raw_dict.get("batch", None)
        non_tensor_batch = raw_dict.get("non_tensor_batch", {}) or {}
        meta_info = raw_dict.get("meta_info", {}) or {}
        if batch is not None and hasattr(batch, "batch_size"):
            data_len = batch.batch_size[0]
        elif non_tensor_batch:
            data_len = len(next(iter(non_tensor_batch.values())))
        else:
            kv_meta = raw_dict.get("_kv_meta", None)
            data_len = len(getattr(kv_meta, "keys", []) or []) if kv_meta is not None else 0
        parts = [
            type(data).__name__,
            f"len={data_len}",
            f"batch={'set' if batch is not None else 'none'}",
            f"tensor_keys={_short_list(batch.keys() if batch is not None else [])}",
            f"non_tensor_keys={_short_list(non_tensor_batch.keys())}",
            f"meta_keys={_short_list(meta_info.keys())}",
        ]
        if isinstance(data, globals().get("LazyDataProto", ())):
            kv_meta = raw_dict.get("_kv_meta", None)
            parts.extend([
                f"lazy_backed={kv_meta is not None}",
                f"materialized={raw_dict.get('_materialized', None)}",
                f"local_cached={_short_list(raw_dict.get('_local_cached_fields', []))}",
                f"local_authoritative={_short_list(raw_dict.get('_local_authoritative_fields', []))}",
            ])
            if kv_meta is not None:
                parts.extend([
                    f"kv_partition={getattr(kv_meta, 'partition_id', None)}",
                    f"kv_key_count={len(getattr(kv_meta, 'keys', []) or [])}",
                    f"kv_fields={_short_list(getattr(kv_meta, 'fields', []) or [])}",
                ])
        return ", ".join(parts)

    if type(data).__name__ == "ObjectRefWrap":
        return f"ObjectRefWrap(collected={getattr(data, 'collected', None)})"

    return type(data).__name__


def log_dataflow(event: str, data=None, **context):
    if not dataflow_debug_enabled():
        return
    ctx = " ".join(f"{key}={value}" for key, value in context.items() if value is not None)
    prefix = f"[dataflow][{event}]"
    if ctx:
        prefix = f"{prefix} {ctx}"
    logger.info(f"{prefix} {describe_dataflow(data)}")


def merge_metrics_into_result(result, metrics: Optional[Dict[str, Any]] = None):
    if not metrics:
        return result
    if isinstance(result, DataProto):
        if result.meta_info is None:
            result.meta_info = {}
        current_metrics = result.meta_info.get("metrics", {}) or {}
        current_metrics.update(metrics)
        result.meta_info["metrics"] = current_metrics
    return result


def pad_dataproto_to_divisor(data: "DataProto", size_divisor: int):
    """Pad a DataProto to size divisible by size_divisor

    Args:
        size_divisor (int): size divisor

    Returns:
        data: (DataProto): the padded DataProto
        pad_size (int)
    """
    assert isinstance(data, DataProto), "data must be a DataProto"
    if len(data) % size_divisor != 0:
        pad_size = size_divisor - len(data) % size_divisor
        padding_protos = []
        remaining_pad = pad_size
        while remaining_pad > 0:
            take_size = min(remaining_pad, len(data))
            padding_protos.append(data[:take_size])
            remaining_pad -= take_size
        data_padded = DataProto.concat([data] + padding_protos)
    else:
        pad_size = 0
        data_padded = data
    return data_padded, pad_size


def unpad_dataproto(data: "DataProto", pad_size):
    if pad_size != 0:
        data = data[:-pad_size]
    return data


def union_tensor_dict(tensor_dict1: TensorDict, tensor_dict2: TensorDict) -> TensorDict:
    """Union two tensordicts."""
    assert (
        tensor_dict1.batch_size == tensor_dict2.batch_size
    ), f"Two tensor dict must have identical batch size. Got {tensor_dict1.batch_size} and {tensor_dict2.batch_size}"
    for key in tensor_dict2.keys():
        if key not in tensor_dict1.keys():
            tensor_dict1[key] = tensor_dict2[key]
        else:
            assert tensor_dict1[key].equal(
                tensor_dict2[key]
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"

    return tensor_dict1


def union_numpy_dict(tensor_dict1: dict[np.ndarray], tensor_dict2: dict[np.ndarray]) -> dict[np.ndarray]:
    for key, val in tensor_dict2.items():
        if key in tensor_dict1:
            assert isinstance(tensor_dict2[key], np.ndarray)
            assert isinstance(tensor_dict1[key], np.ndarray)
            assert np.all(
                tensor_dict2[key] == tensor_dict1[key]
            ), f"{key} in tensor_dict1 and tensor_dict2 are not the same object"
        tensor_dict1[key] = val

    return tensor_dict1


def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
    """
    Convert a list of dictionaries into a dictionary of lists.

    Example:
        Input:  [{"a": 1, "b": 2}, {"a": 3}, {"b": 4}]
        Output: {"a": [1, 3], "b": [2, 4]}

    Only keys present in each dictionary are aggregated.
    Missing keys in a dictionary are simply skipped.
    """
    if not list_of_dict:
        return {}

    output = {}
    for d in list_of_dict:
        if not isinstance(d, dict):
            raise TypeError(f"Expected dict, but got {type(d)}: {d}")
        for k, v in d.items():
            output.setdefault(k, []).append(v)

    return output


def collate_fn(x: list["DataProtoItem"]):
    batch = []
    non_tensor_batch = []
    meta_info = None
    for data in x:
        meta_info = data.meta_info
        batch.append(data.batch)
        non_tensor_batch.append(data.non_tensor_batch)
    batch = torch.stack(batch).contiguous()
    non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
    for key, val in non_tensor_batch.items():
        non_tensor_batch[key] = np.empty(len(val), dtype=object)
        non_tensor_batch[key][:] = val
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


def move_tensors_to_device(data, device):
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = move_tensors_to_device(val, device)
    elif isinstance(data, list):
        for index, val in enumerate(data):
            data[index] = move_tensors_to_device(val, device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def custom_np_concatenate(val):
    concatenated_list = []
    for array in val:
        concatenated_list.extend(array)
    concatenated_array = np.empty(len(concatenated_list), dtype=object)
    concatenated_array[:] = concatenated_list
    return concatenated_array


class _TrackedTensorDictProxy:
    """A thin proxy that preserves DataProto-style batch access while letting
    LazyDataProto track local tensor updates."""

    def __init__(self, owner: "LazyDataProto", td: TensorDict):
        self._owner = owner
        self._td = td

    def __getattr__(self, name):
        return getattr(self._td, name)

    def __getitem__(self, item):
        if isinstance(item, str) and item not in self._td.keys():
            tensor_field_names = set(getattr(self._owner, "_tensor_field_names", []) or [])
            kv_fields = set(getattr(getattr(self._owner, "_kv_meta", None), "fields", None) or [])
            if item in tensor_field_names or item in kv_fields:
                self._owner.materialize(fields=[item])
                self._td = self._owner._raw_batch()
        return self._td[item]

    def __setitem__(self, key, value):
        self._owner.set_tensor_field(key, value)

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self._td)

    def __len__(self):
        return len(self._td)

    def keys(self):
        key_list = list(self._td.keys())
        known = set(key_list)
        for key in getattr(self._owner, "_tensor_field_names", []) or []:
            if key not in known:
                key_list.append(key)
                known.add(key)
        kv_meta = getattr(self._owner, "_kv_meta", None)
        for key in getattr(kv_meta, "fields", None) or []:
            if key not in known:
                key_list.append(key)
                known.add(key)
        return key_list

    def items(self):
        return self._td.items()

    def values(self):
        return self._td.values()

    def pop(self, key, *args):
        value = self._td.pop(key, *args)
        self._owner._local_authoritative_fields.discard(key)
        self._owner._local_cached_fields.discard(key)
        if key in self._owner._tensor_field_names:
            self._owner._tensor_field_names.remove(key)
        if self._owner._kv_meta is not None and hasattr(self._owner._kv_meta, "fields"):
            self._owner._kv_meta.fields = list(dict.fromkeys(self._owner._tensor_field_names))
        self._owner._refresh_materialized_status()
        return value

    def rename_key_(self, old_key, new_key):
        if isinstance(old_key, (tuple, list)) or isinstance(new_key, (tuple, list)):
            old_keys = list(old_key) if isinstance(old_key, (tuple, list)) else [old_key]
            new_keys = list(new_key) if isinstance(new_key, (tuple, list)) else [new_key]
            if len(old_keys) != len(new_keys):
                raise ValueError(
                    f"rename_key_ expects old/new keys with same length, got {len(old_keys)} and {len(new_keys)}"
                )
            for src, dst in zip(old_keys, new_keys, strict=True):
                self.rename_key_(src, dst)
            return self

        if old_key not in self._td.keys():
            tensor_field_names = set(getattr(self._owner, "_tensor_field_names", []) or [])
            kv_fields = set(getattr(getattr(self._owner, "_kv_meta", None), "fields", None) or [])
            if old_key in tensor_field_names or old_key in kv_fields:
                self._owner.materialize(fields=[old_key])
                self._td = self._owner._raw_batch()

        self._td.rename_key_(old_key, new_key)
        if old_key in self._owner._tensor_field_names:
            self._owner._tensor_field_names.remove(old_key)
        if new_key not in self._owner._tensor_field_names:
            self._owner._tensor_field_names.append(new_key)
        if old_key in self._owner._local_authoritative_fields:
            self._owner._local_authoritative_fields.remove(old_key)
            self._owner._local_authoritative_fields.add(new_key)
        if old_key in self._owner._local_cached_fields:
            self._owner._local_cached_fields.remove(old_key)
            self._owner._local_cached_fields.add(new_key)
        if self._owner._kv_meta is not None and hasattr(self._owner._kv_meta, "fields"):
            self._owner._kv_meta.fields = list(dict.fromkeys(self._owner._tensor_field_names))
        self._owner._refresh_materialized_status()
        return self


@dataclass
class DataProtoItem:
    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)


@dataclass
class DataProto:
    """
    A DataProto is a data structure that aims to provide a standard protocol for data exchange between functions.
    It contains a batch (TensorDict) and a meta_info (Dict). The batch is a TensorDict https://pytorch.org/tensordict/.
    TensorDict allows you to manipulate a dictionary of Tensors like a single Tensor. Ideally, the tensors with the
    same batch size should be put inside batch.
    """

    batch: TensorDict = None
    non_tensor_batch: Dict = field(default_factory=dict)
    meta_info: Dict = field(default_factory=dict)

    def __post_init__(self):
        # perform necessary checking
        self.check_consistency()
    
        if self.batch is not None and current_platform.is_npu():
            for key, val in self.batch.items():
                if isinstance(val, torch.Tensor) and val.dtype == torch.int64:
                    logger.debug(f"[NPU] Converting Tensor {key} from int64 -> int32, shape={val.shape}")
                    self.batch[key] = val.to(torch.int32)

    def __len__(self):
        if self.batch is not None:
            return self.batch.batch_size[0]
        if self.non_tensor_batch is not None:
            return len(next(iter(self.non_tensor_batch.values())))
        return 0

    def __getitem__(self, item):
        """
        Enhanced indexing for DataProto objects.

        Args:
            item: Can be one of:
                - int: A single index
                - slice: A slice object (start:stop:step)
                - list: A list of indices
                - numpy.ndarray: An array of indices
                - torch.Tensor: A tensor of indices

        Returns:
            DataProto: For all indexing types except single integers
            DataProtoItem: Only for single integer indices
        """
        # Case 1: Slice object - use the slice method
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)

        # Case 2: List, numpy array, or torch tensor - use sel_idxs
        elif isinstance(item, (list, np.ndarray, torch.Tensor)):
            return self.select_idxs(item)

        # Case 3: Single integer - return DataProtoItem for backward compatibility
        elif isinstance(item, (int, np.integer)):
            tensor_data = self.batch[item]
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return DataProtoItem(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)

        # # Case 4: Unsupported type
        else:
            raise TypeError(f"Indexing with {type(item)} is not supported")

    def __getstate__(self):
        import io

        buffer = io.BytesIO()
        if tensordict.__version__ >= "0.5.0" and self.batch is not None:
            self.batch = self.batch.contiguous()
            self.batch = self.batch.consolidate()
        torch.save(self.batch, buffer)
        return buffer, self.non_tensor_batch, self.meta_info

    def __setstate__(self, data):
        batch_deserialized, non_tensor_batch, meta_info = data
        batch_deserialized.seek(0)
        batch = torch.load(
            batch_deserialized, weights_only=False, map_location="cpu" if not current_platform.is_available() else None
        )
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info

    def check_consistency(self):
        """Check the consistency of the DataProto. Mainly for batch and non_tensor_batch
        We expose this function as a public one so that user can call themselves directly
        """
        if self.batch is not None:
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1"

        if len(self.non_tensor_batch) != 0:
            # TODO: we can actually lift this restriction if needed
            assert len(self.batch.batch_size) == 1, "only support num_batch_dims=1 when non_tensor_batch is not empty."

            batch_size = self.batch.batch_size[0]
            for key, val in self.non_tensor_batch.items():
                assert (
                    isinstance(val, np.ndarray) and val.dtype == object
                ), "data in the non_tensor_batch must be a numpy.array with dtype=object"
                assert (
                    val.shape[0] == batch_size
                ), f"key {key} length {len(val)} is not equal to batch size {batch_size}"

    @classmethod
    def from_single_dict(cls, data: Dict[str, Union[torch.Tensor, np.ndarray]], meta_info=None):
        tensors = {}
        non_tensors = {}

        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if current_platform.is_npu() and val.dtype == torch.int64:
                    logger.debug(f"[NPU] Converting Tensor {key} from int64 -> int32, shape={val.shape}")
                    val = val.to(torch.int32)
                tensors[key] = val
            elif isinstance(val, np.ndarray):
                non_tensors[key] = val
            else:
                raise ValueError(f"Unsupported type in data {type(val)}")

        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @classmethod
    def from_dict(cls, tensors: Dict[str, torch.Tensor], non_tensors=None, meta_info=None, num_batch_dims=1):
        """Create a DataProto from a dict of tensors. This assumes that
        1. All the tensor in tensors have the same dim0
        2. Only dim0 is the batch dim
        """
        assert len(tensors) > 0, "tensors must not be empty"
        assert num_batch_dims > 0, "num_batch_dims must be greater than zero"
        if non_tensors is not None:
            assert num_batch_dims == 1, "only support num_batch_dims=1 when non_tensors is not None."

        if meta_info is None:
            meta_info = {}
        if non_tensors is None:
            non_tensors = {}

        assert isinstance(non_tensors, dict)

        # get and check batch size
        batch_size = None
        pivot_key = None
        for key, tensor in tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[:num_batch_dims]
                pivot_key = key
            else:
                current_batch = tensor.shape[:num_batch_dims]
                assert (
                    batch_size == current_batch
                ), f"Not all the tensor in tensors have the same batch size with batch_dims={num_batch_dims}. Got {pivot_key} has {batch_size}, {key} has {current_batch}"

        for key, val in non_tensors.items():
            non_tensors[key] = np.empty(len(val), dtype=object)
            non_tensors[key][:] = val

        tensor_dict = TensorDict(source=tensors, batch_size=batch_size)
        return cls(batch=tensor_dict, non_tensor_batch=non_tensors, meta_info=meta_info)

    def to(self, device) -> "DataProto":
        """move the batch to device

        Args:
            device (torch.device, str): torch device

        Returns:
            DataProto: the current DataProto

        """
        if self.batch is not None:
            self.batch = self.batch.to(device)
        if self.meta_info is not None:
            self.meta_info = move_tensors_to_device(self.meta_info, device)

        return self

    def clone(self) -> "DataProto":
        """
        Create a deep copy of this DataProto, including tensors,
        non-tensor data, and meta_info.

        The new DataProto will share no underlying storage with the original.

        Returns:
            DataProto: A new DataProto instance with the same content but
                       independent memory.
        """
        # Copy batch
        batch_copy = self.batch.clone() if self.batch is not None else None

        # Copy non-tensor objects (numpy arrays)
        non_tensor_copy = {k: np.copy(v) for k, v in self.non_tensor_batch.items()}

        # Deep copy meta_info to avoid shared mutable objects
        meta_copy = copy.deepcopy(self.meta_info)

        # Return new DataProto instance
        return DataProto(
            batch=batch_copy,
            non_tensor_batch=non_tensor_copy,
            meta_info=meta_copy
        )

    def select(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None, deepcopy=False) -> "DataProto":
        """Select a subset of the DataProto via batch_keys and meta_info_keys

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to select
            meta_info_keys (list, optional): a list of keys indicating the meta info to select

        Returns:
            DataProto: the DataProto with the selected batch_keys and meta_info_keys
        """
        if batch_keys is not None:
            batch_keys = tuple(batch_keys)
            sub_batch = self.batch.select(*batch_keys)
        else:
            sub_batch = self.batch

        if non_tensor_batch_keys is not None:
            non_tensor_batch = {key: val for key, val in self.non_tensor_batch.items() if key in non_tensor_batch_keys}
        else:
            non_tensor_batch = self.non_tensor_batch

        if deepcopy:
            non_tensor_batch = copy.deepcopy(non_tensor_batch)

        if meta_info_keys is not None:
            sub_meta_info = {key: val for key, val in self.meta_info.items() if key in meta_info_keys}
        else:
            sub_meta_info = self.meta_info

        if deepcopy:
            sub_meta_info = copy.deepcopy(sub_meta_info)

        return DataProto(batch=sub_batch, non_tensor_batch=non_tensor_batch, meta_info=sub_meta_info)

    def select_idxs(self, idxs):
        """
        Select specific indices from the DataProto.

        Args:
            idxs (torch.Tensor or numpy.ndarray or list): Indices to select

        Returns:
            DataProto: A new DataProto containing only the selected indices
        """
        if isinstance(idxs, list):
            idxs = torch.tensor(idxs)
            if idxs.dtype != torch.bool:
                idxs = idxs.type(torch.int32)

        if isinstance(idxs, np.ndarray):
            idxs_np = idxs
            idxs_torch = torch.from_numpy(idxs)
        else:  # torch.Tensor
            idxs_torch = idxs
            idxs_np = idxs.detach().cpu().numpy()

        batch_size = idxs_np.sum() if idxs_np.dtype == bool else idxs_np.shape[0]

        if self.batch is not None:
            # Use TensorDict's built-in indexing capabilities
            selected_batch = TensorDict(
                source={key: tensor[idxs_torch] for key, tensor in self.batch.items()}, batch_size=(batch_size,)
            )
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            selected_non_tensor[key] = val[idxs_np]

        return type(self)(batch=selected_batch, non_tensor_batch=selected_non_tensor, meta_info=self.meta_info)

    def slice(self, start=None, end=None, step=None):
        """
        Slice the DataProto and return a new DataProto object.
        This is an improved version of direct slicing which returns a DataProtoItem.

        Args:
            start (int, optional): Start index. Defaults to None (start from beginning).
            end (int, optional): End index (exclusive). Defaults to None (go to end).
            step (int, optional): Step size. Defaults to None (step=1).

        Returns:
            DataProto: A new DataProto containing the sliced data

        Examples:
            # Using the slice method directly
            sliced_data = data_proto.slice(10, 20)

            # Using enhanced indexing (returns DataProto)
            sliced_data = data_proto[10:20]
            sliced_data = data_proto[::2]  # Every other element

            # Using list indexing (returns DataProto)
            indices = [1, 5, 10]
            selected_data = data_proto[indices]

            # Single index still returns DataProtoItem
            single_item = data_proto[5]
        """
        # Create a slice object
        slice_obj = slice(start, end, step)

        # Handle the batch data
        if self.batch is not None:
            # Use TensorDict's built-in slicing capabilities
            sliced_batch = self.batch[slice_obj]
        else:
            sliced_batch = None

        # Handle the non-tensor batch data
        sliced_non_tensor = {}
        for key, val in self.non_tensor_batch.items():
            sliced_non_tensor[key] = val[slice_obj]

        # Return a new DataProto object
        return type(self)(batch=sliced_batch, non_tensor_batch=sliced_non_tensor, meta_info=self.meta_info)

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        """Pop a subset of the DataProto via `batch_keys` and `meta_info_keys`

        Args:
            batch_keys (list, optional): a list of strings indicating the keys in batch to pop
            meta_info_keys (list, optional): a list of keys indicating the meta info to pop

        Returns:
            DataProto: the DataProto with the poped batch_keys and meta_info_keys
        """
        assert batch_keys is not None
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []
        batch_keys = self.validate_input(batch_keys)
        non_tensor_batch_keys = self.validate_input(non_tensor_batch_keys)
        meta_info_keys = self.validate_input(meta_info_keys)

        tensors = {}
        # tensor batch
        for key in batch_keys:
            assert key in self.batch.keys()
            tensors[key] = self.batch.pop(key)
        non_tensors = {}
        # non tensor batch
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)
        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)
        return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)

    @staticmethod
    def validate_input(keys):
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            elif isinstance(keys, list):
                pass
            else:
                raise TypeError(f"keys must be a list or a string, but got {type(keys)}")
        return keys

    def rename(self, old_keys=None, new_keys=None) -> "DataProto":
        """
        Note that this function only rename the key in the batch
        """

        old_keys = self.validate_input(old_keys)
        new_keys = self.validate_input(new_keys)

        if len(new_keys) != len(old_keys):
            raise ValueError(
                f"new_keys and old_keys must have the same length, but got {len(new_keys)} and {len(old_keys)}"
            )

        self.batch.rename_key_(tuple(old_keys), tuple(new_keys))

        return self

    def union(self, other: "DataProto") -> "DataProto":
        """Union with another DataProto. Union batch and meta_info separately.
        Throw an error if
        - there are conflict keys in batch and they are not equal
        - the batch size of two data batch is not the same
        - there are conflict keys in meta_info and they are not the same.

        Args:
            other (DataProto): another DataProto to union

        Returns:
            DataProto: the DataProto after union
        """
        self.batch = union_tensor_dict(self.batch, other.batch)
        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other.meta_info)
        return self

    def make_iterator(self, mini_batch_size, epochs, seed=None, dataloader_kwargs=None):
        """Make an iterator from the DataProto. This is built upon that TensorDict can be used as a normal Pytorch
        dataset. See https://pytorch.org/tensordict/tutorials/data_fashion for more details.

        Args:
            mini_batch_size (int): mini-batch size when iterating the dataset. We require that
                ``batch.batch_size[0] % mini_batch_size == 0``
            epochs (int): number of epochs when iterating the dataset.
            dataloader_kwargs: internally, it returns a DataLoader over the batch.
                The dataloader_kwargs is the kwargs passed to the DataLoader

        Returns:
            Iterator: an iterator that yields a mini-batch data at a time. The total number of iteration steps is
            ``self.batch.batch_size * epochs // mini_batch_size``
        """
        assert self.batch.batch_size[0] % mini_batch_size == 0, f"{self.batch.batch_size[0]} % {mini_batch_size} != 0"
        # we can directly create a dataloader from TensorDict
        if dataloader_kwargs is None:
            dataloader_kwargs = {}

        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = None

        assert isinstance(dataloader_kwargs, Dict)
        train_dataloader = DataLoader(
            dataset=self, batch_size=mini_batch_size, collate_fn=collate_fn, generator=generator, **dataloader_kwargs
        )

        def get_data():
            for _ in range(epochs):
                for d in train_dataloader:
                    d.meta_info = self.meta_info
                    yield d

        return iter(get_data())

    def chunk(self, chunks: int) -> List["DataProto"]:
        """Split the batch among dim=0 into chunks. The meta_info is passed to each DataProto after split.
        要求:
            batch_size > chunks，调用方保证，此处保证每个chunk会返回一个DataProto

        np.array_split(val, chunks) 和 self.batch.chunk(chunks=chunks, dim=0) 在不能均分时行为不同
        Args:
            chunks (int): the number of chunks to split on dim=0

        Returns:
            List[DataProto]: a list of DataProto after splitting
        """
        chunks_sizes = None
        if len(self) > 0:
            assert len(self) >= chunks, f"batch_size {self.batch.batch_size[0]} < chunks {chunks}"
            index_array = np.arange(len(self))
            chunks_sizes = [len(b) for b in np.array_split(index_array, chunks)]

        if self.batch is not None:
            batch_lst = divide_by_chunk_size(self.batch, chunk_sizes=chunks_sizes)
        else:
            batch_lst = [None for _ in range(chunks)]

        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            assert isinstance(val, np.ndarray)
            non_tensor_lst = divide_by_chunk_size(val, chunk_sizes=chunks_sizes)
            assert len(non_tensor_lst) == chunks, f"len(non_tensor_lst) {len(non_tensor_lst)} != chunks {chunks}"
            for i in range(chunks):
                non_tensor_batch_lst[i][key] = non_tensor_lst[i]

        output = []
        for i in range(chunks):
            output.append(
                DataProto(
                    batch=batch_lst[i].clone() if batch_lst[i] is not None else batch_lst[i],
                    non_tensor_batch=non_tensor_batch_lst[i],
                    meta_info=self.meta_info,
                )
            )

        return output

    @staticmethod
    def concat(
            data: List["DataProto"],
            *,
            global_keys: Optional[Set[str]] = None,
    ) -> "DataProto":
        """
        Concatenate a list of DataProto objects.

        Parameters
        ----------
        data : List[DataProto]
            List of DataProto instances to be concatenated.
        global_keys : Set[str], optional
            Keys in `meta_info` that should be **aggregated across ranks**.
            - If the value is a dict, each sub-key is concatenated across ranks.
            - Otherwise, values are collected into a list.
            Keys not listed retain only the value from rank 0.

        Returns
        -------
        DataProto
            A new DataProto with concatenated tensors, non-tensor data,
            and processed meta information.
        """
        global_keys = global_keys if global_keys is not None else {"metrics"}
        log_dataflow("protocol.dataproto.concat.enter", data, global_keys=sorted(global_keys))

        lazy_cls = globals().get("LazyDataProto")

        def _as_eager_dataproto(item):
            if lazy_cls is not None and isinstance(item, lazy_cls):
                return item.to_dataproto()
            return item

        if lazy_cls is not None and any(isinstance(d, lazy_cls) for d in data):
            if all(isinstance(d, lazy_cls) for d in data):
                log_dataflow("protocol.dataproto.concat.delegate_lazy", data)
                return lazy_cls.concat(data, global_keys=global_keys)
            #data = [d.materialize() if isinstance(d, lazy_cls) else d for d in data]
            data = [_as_eager_dataproto(d) for d in data]
            log_dataflow("protocol.dataproto.concat.normalized_eager", data)

        # ---------- 1. Concatenate tensor / non-tensor batches ----------
        batch_lst = []
        for d in data:
            batch = d.batch
            if batch is None:
                continue
            if isinstance(batch, _TrackedTensorDictProxy):
                batch = batch._td
            batch_lst.append(batch)
        new_batch = torch.cat(batch_lst, dim=0) if batch_lst else None

        non_tensor_batch = list_of_dict_to_dict_of_list(
            [d.non_tensor_batch for d in data]
        )
        for k, v in non_tensor_batch.items():
            non_tensor_batch[k] = custom_np_concatenate(v)

        # ---------- 2. Aggregate meta information ----------
        merged_meta = dict(data[0].meta_info)  # start with rank-0 values

        for key in global_keys:
            if key not in merged_meta:
                continue

            values = [d.meta_info.get(key) for d in data]

            # Case 1: dict — aggregate each sub-key across ranks
            if isinstance(merged_meta[key], dict):
                sub_dict = list_of_dict_to_dict_of_list(values)
                for sub_key, sub_list in sub_dict.items():
                    try:
                        if np.isscalar(sub_list[0]):
                            sub_dict[sub_key] = np.array(sub_list).tolist()
                        else:
                            sub_dict[sub_key] = np.concatenate(sub_list, axis=0).tolist()
                    except Exception:
                        # fallback: keep as list
                        sub_dict[sub_key] = sub_list
                merged_meta[key] = sub_dict

            # Case 2: non-dict — collect into list
            else:
                merged_meta[key] = values

        result = DataProto(
            batch=new_batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=merged_meta,
        )
        log_dataflow("protocol.dataproto.concat.exit", result)
        return result

    def reorder(self, indices):
        """
        Note that this operation is in-place
        """
        # Ensure that indices is at least a 1-D tensor.
        indices = indices.view(-1) if indices.dim() == 0 else indices
        indices_np = indices.detach().numpy()
        self.batch = self.batch[indices]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}

    def group_by(self, keys: Union[List[str], str]) -> Dict[str, "DataProto"]:
        """
        Group the data by specified keys. Supports grouping by both tensor and non-tensor fields.

        Args:
            keys: Field names to group by. Can be either in batch (tensors) or non_tensor_batch

        Returns:
            Dictionary mapping group keys to DataProto instances containing matching data

        Example:
            Given data with field "category" having values ["A", "B", "A"],
            returns {"A": DataProto(A_data), "B": DataProto(B_data)}
        """
        keys = self.validate_input(keys)
        assert len(keys) > 0, "Must provide at least one grouping key"

        # Collect grouping values across data types
        group_key_values = []
        for idx in range(len(self)):
            key_values = []
            for key in keys:
                # Check tensor data first
                if key in self.batch.keys():
                    key_values.append(str(self.batch[key][idx].numpy()))
                elif key in self.non_tensor_batch:
                    key_values.append(str(self.non_tensor_batch[key][idx]))
                else:
                    raise KeyError(f"Grouping key '{key}' not found in tensor or non-tensor data")

            # Create composite key for multi-field grouping
            group_key = "|".join(key_values) if len(key_values) > 1 else key_values[0]
            group_key_values.append(group_key)

        # Create index groups
        groups = defaultdict(list)
        for idx, group_key in enumerate(group_key_values):
            groups[group_key].append(idx)

        # Create grouped DataProtos
        grouped_data = {}
        for group_key, indices in groups.items():
            grouped_data[group_key] = collate_fn([self[idx] for idx in indices])

        return grouped_data

    def repeat(self, repeat_times=2, interleave=True):
        """
        Repeat the batch data a specified number of times.

        Args:
            repeat_times (int): Number of times to repeat the data.
            interleave (bool): Whether to interleave the repeated data.

        Returns:
            DataProto: A new DataProto with repeated data.
        """
        if self.batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0) for key, tensor in self.batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0).expand(repeat_times, *tensor.shape).reshape(-1, *tensor.shape[1:])
                    for key, tensor in self.batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(self.batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in self.non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(val, (repeat_times,) + (1,) * (val.ndim - 1))

        return type(self)(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=self.meta_info,
        )

    @staticmethod
    def materialize_concat(
            data_refs: Union[List[ray.ObjectRef], ray.ObjectRef, List["ObjectRefWrap"]],
            *,
            global_keys: Optional[Set[str]] = None,
    ) -> "DataProto":
        """
        Fetch a collection of DataProto objects from Ray ObjectRef(s) and concatenate
        them into a single DataProto instance.

        Parameters
        ----------
        data_refs : Union[List[ray.ObjectRef], ray.ObjectRef, List[ObjectRefWrap]]
            Ray object references (or ObjectRefWrap) pointing to DataProto objects.
        global_keys : Optional[Set[str]], optional
            Keys in ``meta_info`` that should be aggregated across all ranks when
            concatenating.  If None, only rank-0 values are kept for all keys.

        Returns
        -------
        DataProto
            The concatenated DataProto instance.
        """
        log_dataflow(
            "protocol.dataproto.materialize_concat.enter",
            data_refs,
            global_keys=sorted(global_keys or {"metrics"}),
        )
        # Normalize input to List[<reference>]
        if isinstance(data_refs, (DataProto, LazyDataProto)):
            data_refs = [data_refs]
        elif not isinstance(data_refs, (list, tuple)):
            data_refs = [data_refs]

        timeout = None
        if "roll_RPC_TIMEOUT" in os.environ:
            timeout = int(os.environ["roll_RPC_TIMEOUT"])

        # Fetch objects from Ray
        if isinstance(data_refs[0], ObjectRefWrap):
            data_refs: List[ObjectRefWrap]
            obj_refs = [ref.obj_ref for ref in data_refs]
            fetched = ray.get(obj_refs, timeout=timeout)
            data = [fetched[i] for i, ref in enumerate(data_refs) if ref.collected]
        elif isinstance(data_refs[0], ray.ObjectRef):
            data = ray.get(data_refs, timeout=timeout)
        else:
            data = list(data_refs)

        BatchMeta, KVBatchMeta = BatchData._maybe_import_tq_types()
        if BatchMeta is not None:
            from roll.utils.transferqueue_utils import meta_to_dataproto

            normalized = []
            for item in data:
                if isinstance(item, (BatchMeta, KVBatchMeta)):
                    normalized.append(meta_to_dataproto(item))
                elif isinstance(item, LazyDataProto):
                    normalized.append(item.materialize())
                else:
                    normalized.append(item)
            data = normalized

        # Concatenate and apply global aggregation rules
        result = DataProto.concat(data, global_keys=global_keys)
        log_dataflow("protocol.dataproto.materialize_concat.exit", result)
        return result


@dataclass
class LazyDataProto(DataProto):
    """A minimal TQ-backed DataProto variant.

    This class keeps the public DataProto surface while delegating actual tensor
    storage to TransferQueue through KVBatchMeta. For now we only implement the
    minimum protocol pieces needed by BatchData / dispatch / collect:
    - `materialize(fields=None)` to cache tensor fields locally while keeping
      lazy/TQ-backed identity
    - `to_dataproto()` to explicitly detach into eager DataProto
    - `chunk(chunks)`
    - `concat(data_list)`

    Deeper lazy behaviors such as lazy `group_by/select/reorder` can be added
    incrementally after the transport path is stable.
    """

    _kv_meta: Any = field(default=None, repr=False, init=False)
    _materialized: bool = field(default=True, repr=False, init=False)
    _tensor_field_names: List[str] = field(default_factory=list, repr=False, init=False)
    _local_authoritative_fields: Set[str] = field(default_factory=set, repr=False, init=False)
    _local_cached_fields: Set[str] = field(default_factory=set, repr=False, init=False)
    _materialize_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, init=False)

    def check_consistency(self):
        if self.batch is not None:
            return DataProto.check_consistency(self)

        if len(self.non_tensor_batch) != 0:
            batch_size = len(next(iter(self.non_tensor_batch.values())))
            for key, val in self.non_tensor_batch.items():
                assert (
                    isinstance(val, np.ndarray) and val.dtype == object
                ), "data in non_tensor_batch must be numpy.ndarray with dtype=object"
                assert (
                    val.shape[0] == batch_size
                ), f"key {key} length {len(val)} is not equal to batch size {batch_size}"

    def __len__(self):
        raw_batch = self._raw_batch()
        if raw_batch is not None:
            return raw_batch.batch_size[0]
        if self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
            return len(next(iter(self.non_tensor_batch.values())))
        if self._kv_meta is not None:
            return len(self._kv_meta.keys)
        return 0

    def __getattribute__(self, name):
        if name == "batch":
            try:
                kv_meta = object.__getattribute__(self, "_kv_meta")
                materialized = object.__getattribute__(self, "_materialized")
                raw_dict = object.__getattribute__(self, "__dict__")
                batch = raw_dict.get("batch", None)
                if batch is not None:
                    return _TrackedTensorDictProxy(self, batch)
                if kv_meta is not None and not materialized:
                    if self.non_tensor_batch is not None and len(self.non_tensor_batch) > 0:
                        batch_size = len(next(iter(self.non_tensor_batch.values())))
                    else:
                        batch_size = len(kv_meta.keys)
                    return _TrackedTensorDictProxy(self, TensorDict({}, batch_size=(batch_size,)))
            except AttributeError:
                pass
        return object.__getattribute__(self, name)

    def __getstate__(self):
        import io

        raw_batch = object.__getattribute__(self, "__dict__").get("batch", None)
        buffer = None
        if raw_batch is not None:
            if tensordict.__version__ >= "0.5.0":
                raw_batch = raw_batch.contiguous()
                raw_batch = raw_batch.consolidate()
                object.__setattr__(self, "batch", raw_batch)
            buffer = io.BytesIO()
            torch.save(raw_batch, buffer)

        return {
            "batch_buffer": buffer,
            "non_tensor_batch": self.non_tensor_batch,
            "meta_info": self.meta_info,
            "kv_meta": object.__getattribute__(self, "__dict__").get("_kv_meta", None),
            "materialized": object.__getattribute__(self, "__dict__").get("_materialized", True),
            "tensor_field_names": list(object.__getattribute__(self, "__dict__").get("_tensor_field_names", [])),
            "local_authoritative_fields": set(
                object.__getattribute__(self, "__dict__").get("_local_authoritative_fields", set())
            ),
            "local_cached_fields": set(
                object.__getattribute__(self, "__dict__").get("_local_cached_fields", set())
            ),
        }

    def __setstate__(self, data):
        batch_buffer = data.get("batch_buffer")
        batch = None
        if batch_buffer is not None:
            batch_buffer.seek(0)
            batch = torch.load(
                batch_buffer,
                weights_only=False,
                map_location="cpu" if not current_platform.is_available() else None,
            )
        self.batch = batch
        self.non_tensor_batch = data.get("non_tensor_batch", {})
        self.meta_info = data.get("meta_info", {})
        self._kv_meta = data.get("kv_meta")
        self._materialized = data.get("materialized", False)
        self._tensor_field_names = list(data.get("tensor_field_names", []))
        self._local_authoritative_fields = set(data.get("local_authoritative_fields", set()))
        self._local_cached_fields = set(data.get("local_cached_fields", set()))
        self._materialize_lock = threading.Lock()

    def _refresh_materialized_status(self) -> None:
        current_batch = object.__getattribute__(self, "__dict__").get("batch", None)
        current_keys = set(current_batch.keys()) if current_batch is not None else set()
        all_tensor_fields = list(dict.fromkeys(self._tensor_field_names or getattr(self._kv_meta, "fields", None) or []))
        object.__setattr__(self, "_materialized", bool(all_tensor_fields) and all(field in current_keys for field in all_tensor_fields))

    def _raw_batch(self):
        return object.__getattribute__(self, "__dict__").get("batch", None)

    def _local_overlay_field_names(self) -> Set[str]:
        current_batch = self._raw_batch()
        if current_batch is None:
            return set()
        kv_fields = set(getattr(self._kv_meta, "fields", None) or [])
        overlay_fields = set(self._local_authoritative_fields)
        overlay_fields.update(set(current_batch.keys()) - kv_fields)
        return overlay_fields

    @classmethod
    def from_kv_batch_meta(
        cls,
        kv_meta,
        *,
        tensor_field_names: Optional[List[str]] = None,
        non_tensor_batch: Optional[Dict] = None,
        meta_info: Optional[Dict] = None,
    ) -> "LazyDataProto":
        instance = cls(
            batch=None,
            non_tensor_batch=non_tensor_batch or {},
            meta_info=copy.deepcopy(meta_info or getattr(kv_meta, "extra_info", {}) or {}),
        )
        instance._kv_meta = kv_meta
        instance._materialized = False
        instance._tensor_field_names = list(
            tensor_field_names if tensor_field_names is not None else (getattr(kv_meta, "fields", None) or [])
        )
        instance._local_authoritative_fields = set()
        instance._local_cached_fields = set()
        instance._materialize_lock = threading.Lock()
        return instance

    def materialize(self, fields: Optional[List[str]] = None) -> "LazyDataProto":
        if self._kv_meta is None:
            return self

        with self._materialize_lock:
            import transfer_queue as tq

            tq.init()
            fetch_fields = list(fields or self._tensor_field_names or getattr(self._kv_meta, "fields", None) or [])
            if not fetch_fields:
                return self

            current_batch = self._raw_batch()
            current_keys = set(current_batch.keys()) if current_batch is not None else set()
            missing_fields = [field for field in fetch_fields if field not in current_keys]

            if not missing_fields:
                self._refresh_materialized_status()
                return self

            log_dataflow("protocol.lazy.materialize.enter", self, fields=missing_fields)
            td = tq.kv_batch_get(
                    keys=self._kv_meta.keys,
                    partition_id=self._kv_meta.partition_id,
                    fields=missing_fields,
                )
            if current_batch is None:
                object.__setattr__(self, "batch", td)
                current_keys = set(td.keys())
            else:
                for key in td.keys():
                    current_batch[key] = td[key]
                    current_keys.add(key)

            self._local_cached_fields.update(missing_fields)
            self._refresh_materialized_status()
            log_dataflow("protocol.lazy.materialize.exit", self, fields=missing_fields)
        return self

    def to_dataproto(self) -> "DataProto":
        self.materialize()
        raw_batch = self._raw_batch()
        return DataProto(
            batch=raw_batch,
            non_tensor_batch=self.non_tensor_batch,
            meta_info=self.meta_info,
        )

    def _can_use_lazy_meta_ops(self) -> bool:
        return self._kv_meta is not None and len(self._local_authoritative_fields) == 0

    def _flush_authoritative_to_tq(self) -> None:
        """Flush local-authoritative tensor fields to TQ so that kv_meta is
        up-to-date and _local_authoritative_fields can be cleared.

        This is called automatically before any operation that needs all items
        to be TQ-backed (e.g. concat), so that pipelines never have to call
        prepare_for_remote() manually just to keep lazy identity.

        After the call:
        - Fields successfully written to TQ are removed from _local_authoritative_fields.
        - Fields that TQ rejected (e.g. dtype mismatch) remain in _local_authoritative_fields
          so they can be carried as overlay through concat/select.
        - The local batch cache is kept intact (no extra fetch needed on read).
        """
        if self._kv_meta is None or not self._local_authoritative_fields:
            return
        start_time = time.perf_counter()
        from roll.utils.transferqueue_utils import kv_batch_meta_put_tensordict
        import transfer_queue as tq

        local = self._raw_batch()
        if local is None:
            self._local_authoritative_fields.clear()
            return
        dirty = list(self._local_authoritative_fields)
        present = [f for f in dirty if f in local.keys()]
        if present:
            fields_before = set(getattr(self._kv_meta, "fields", None) or [])
            td = local.select(*present)
            overlap_fields = [field for field in present if field in fields_before]
            if overlap_fields:
                try:
                    existing_td = tq.kv_batch_get(
                        keys=self._kv_meta.keys,
                        partition_id=self._kv_meta.partition_id,
                        fields=overlap_fields,
                    )
                    for field in overlap_fields:
                        if field not in existing_td.keys():
                            continue
                        existing_value = existing_td[field]
                        incoming_value = td[field]
                        existing_dtype = getattr(existing_value, "dtype", None)
                        incoming_dtype = getattr(incoming_value, "dtype", None)
                        if existing_dtype is None or incoming_dtype is None or existing_dtype == incoming_dtype:
                            continue
                        casted_value = incoming_value.to(dtype=existing_dtype)
                        td[field] = casted_value
                        local[field] = casted_value
                except Exception as exc:
                    logger.warning(f"Failed to align dirty field dtype before TQ flush: {exc}")
            self._kv_meta = kv_batch_meta_put_tensordict(
                self._kv_meta, td, "LazyDataProto._flush_authoritative_to_tq"
            )
            # NOTE: do NOT overwrite kv_meta.fields here.
            # kv_batch_meta_put_tensordict already updates meta.fields to reflect
            # what TQ actually accepted. Overwriting would incorrectly add fields
            # that TQ rejected (e.g. dtype mismatch), causing KeyErrors on fetch.
            fields_after = set(getattr(self._kv_meta, "fields", None) or [])
            confirmed = set(present) & fields_after
            failed = set(present) - confirmed
            # Clear authoritative only for fields confirmed in TQ.
            self._local_authoritative_fields -= confirmed
            # Fields that TQ rejected remain in _local_authoritative_fields;
            # they will be carried as overlay through concat.
            for field in confirmed:
                if field not in self._tensor_field_names:
                    self._tensor_field_names.append(field)
            log_dataflow("protocol.lazy.flush_authoritative", self,
                         flushed_fields=list(confirmed), failed_fields=list(failed))
            logger.info(
                "[perf][protocol.lazy.flush_authoritative] "
                f"elapsed={time.perf_counter() - start_time:.6f}s "
                f"dirty={len(dirty)} flushed={len(confirmed)} failed={len(failed)}"
            )
        else:
            self._local_authoritative_fields.clear()
            logger.info(
                "[perf][protocol.lazy.flush_authoritative] "
                f"elapsed={time.perf_counter() - start_time:.6f}s dirty=0 flushed=0 failed=0"
            )


    def set_tensor_field(self, key: str, value: torch.Tensor) -> "LazyDataProto":
        raw_batch = self._raw_batch()
        if raw_batch is None:
            object.__setattr__(self, "batch", TensorDict({}, batch_size=value.shape[:1]))
            raw_batch = self._raw_batch()
        raw_batch[key] = value
        self._local_authoritative_fields.add(key)
        self._local_cached_fields.add(key)
        if key not in self._tensor_field_names:
            self._tensor_field_names.append(key)
        self._refresh_materialized_status()
        return self

    def mark_local_updated(self, fields: Union[str, List[str], Set[str]]) -> "LazyDataProto":
        if isinstance(fields, str):
            fields = [fields]
        self._local_authoritative_fields.update(fields)
        for field_name in fields:
            if field_name not in self._tensor_field_names:
                self._tensor_field_names.append(field_name)
        return self

    def pop(self, batch_keys=None, non_tensor_batch_keys=None, meta_info_keys=None) -> "DataProto":
        assert batch_keys is not None
        if meta_info_keys is None:
            meta_info_keys = []
        if non_tensor_batch_keys is None:
            non_tensor_batch_keys = []
        batch_keys = self.validate_input(batch_keys)
        non_tensor_batch_keys = self.validate_input(non_tensor_batch_keys)
        meta_info_keys = self.validate_input(meta_info_keys)

        if batch_keys:
            missing_local_fields = [
                key
                for key in batch_keys
                if key not in (set(self.batch.keys()) if self.batch is not None else set())
            ]
            if missing_local_fields:
                self.materialize(fields=missing_local_fields)

        log_dataflow(
            "protocol.lazy.pop.enter",
            self,
            batch_keys=batch_keys,
            non_tensor_batch_keys=non_tensor_batch_keys,
            meta_info_keys=meta_info_keys,
        )

        tensors = {}
        raw_batch = self._raw_batch()
        for key in batch_keys:
            if raw_batch is None or key not in raw_batch.keys():
                raise KeyError(f"Tensor key '{key}' not found in LazyDataProto.pop")
            tensors[key] = raw_batch.pop(key)
            if key in self._tensor_field_names:
                self._tensor_field_names.remove(key)
            self._local_authoritative_fields.discard(key)
            self._local_cached_fields.discard(key)

        non_tensors = {}
        for key in non_tensor_batch_keys:
            assert key in self.non_tensor_batch.keys()
            non_tensors[key] = self.non_tensor_batch.pop(key)

        meta_info = {}
        for key in meta_info_keys:
            assert key in self.meta_info.keys()
            meta_info[key] = self.meta_info.pop(key)

        if self._kv_meta is not None and hasattr(self._kv_meta, "fields"):
            self._kv_meta.fields = list(dict.fromkeys(self._tensor_field_names))

        self._refresh_materialized_status()

        result = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)
        log_dataflow("protocol.lazy.pop.exit", self, remaining_tensor_fields=list(self._tensor_field_names))
        return result

    def select(
        self,
        batch_keys=None,
        non_tensor_batch_keys=None,
        meta_info_keys=None,
        deepcopy=False,
    ) -> "LazyDataProto":
        """Lazy-aware select: returns a LazyDataProto narrowed to the requested
        fields instead of triggering a full materialize.

        When kv_meta is present, we only update ``_tensor_field_names`` so that
        a later materialize() fetches only the requested subset from TQ.  Local
        cache entries that are already present are inherited directly.

        Falls back to the parent DataProto.select when there is no kv_meta
        (i.e. the object is already fully eager).
        """
        if self._kv_meta is None:
            return DataProto.select(
                self,
                batch_keys=batch_keys,
                non_tensor_batch_keys=non_tensor_batch_keys,
                meta_info_keys=meta_info_keys,
                deepcopy=deepcopy,
            )

        # --- tensor fields ---
        all_known = list(dict.fromkeys(self._tensor_field_names))
        if batch_keys is not None:
            batch_keys = list(batch_keys)
            new_tensor_fields = [f for f in batch_keys if f in all_known]
        else:
            new_tensor_fields = all_known

        # --- non-tensor fields ---
        if non_tensor_batch_keys is not None:
            new_non_tensor = {
                k: (copy.deepcopy(v) if deepcopy else v)
                for k, v in self.non_tensor_batch.items()
                if k in non_tensor_batch_keys
            }
        else:
            new_non_tensor = (
                copy.deepcopy(self.non_tensor_batch) if deepcopy else dict(self.non_tensor_batch)
            )

        # --- meta info ---
        if meta_info_keys is not None:
            new_meta = {k: v for k, v in self.meta_info.items() if k in meta_info_keys}
        else:
            new_meta = self.meta_info
        if deepcopy:
            new_meta = copy.deepcopy(new_meta)

        # Build new LazyDataProto with narrowed field list
        new_kv_meta = copy.deepcopy(self._kv_meta)
        if hasattr(new_kv_meta, "fields"):
            new_kv_meta.fields = list(new_tensor_fields)

        result = type(self).from_kv_batch_meta(
            new_kv_meta,
            tensor_field_names=new_tensor_fields,
            non_tensor_batch=new_non_tensor,
            meta_info=new_meta,
        )

        # Inherit local cache / authoritative entries that are still relevant
        local = self._raw_batch()
        if local is not None:
            present = {f: local[f] for f in new_tensor_fields if f in local.keys()}
            if present:
                object.__setattr__(
                    result,
                    "batch",
                    TensorDict(source=present, batch_size=local.batch_size),
                )
                result._local_cached_fields.update(
                    set(present.keys()) & self._local_cached_fields
                )
                result._local_authoritative_fields.update(
                    set(present.keys()) & self._local_authoritative_fields
                )
        result._refresh_materialized_status()
        log_dataflow("protocol.lazy.select.exit", result, batch_keys=new_tensor_fields)
        return result

    def union(self, other: "DataProto") -> "DataProto":
        if not isinstance(other, (DataProto, LazyDataProto)):
            raise TypeError(f"union expects DataProto/LazyDataProto, got {type(other)}")
        if len(self) != len(other):
            raise ValueError(f"Cannot union batches with different lengths: {len(self)} != {len(other)}")

        log_dataflow("protocol.lazy.union.enter", [self, other])

        other_dp = other.to_dataproto() if isinstance(other, LazyDataProto) else other
        if other_dp.batch is not None and len(other_dp.batch.keys()) > 0:
            if self._kv_meta is None:
                return DataProto.union(self, other_dp)

            from roll.utils.transferqueue_utils import kv_batch_meta_put_tensordict

            existing_tensor_fields = set(self._tensor_field_names)
            existing_tensor_fields.update(set(getattr(self._kv_meta, "fields", None) or []))
            raw_batch = self._raw_batch()
            if raw_batch is not None:
                existing_tensor_fields.update(set(raw_batch.keys()))

            incoming_keys = list(other_dp.batch.keys())
            new_tensor_keys = [key for key in incoming_keys if key not in existing_tensor_fields]

            if new_tensor_keys:
                td_to_put = other_dp.batch.select(*new_tensor_keys)
                self._kv_meta = kv_batch_meta_put_tensordict(
                    self._kv_meta,
                    td_to_put,
                    func_name="LazyDataProto.union",
                )
                if raw_batch is not None:
                    for key in td_to_put.keys():
                        raw_batch[key] = td_to_put[key]
                        self._local_cached_fields.add(key)
                incoming_keys = list(td_to_put.keys())
            else:
                incoming_keys = []

            for key in incoming_keys:
                if key not in self._tensor_field_names:
                    self._tensor_field_names.append(key)
                self._local_authoritative_fields.discard(key)
            if hasattr(self._kv_meta, "fields"):
                self._kv_meta.fields = list(dict.fromkeys(self._tensor_field_names))

        self.non_tensor_batch = union_numpy_dict(self.non_tensor_batch, other_dp.non_tensor_batch)
        self.meta_info = union_two_dict(self.meta_info, other_dp.meta_info)
        self._refresh_materialized_status()
        log_dataflow("protocol.lazy.union.exit", self)
        return self

    def prepare_for_remote(self, required_fields: Optional[List[str]] = None):
        """Sync local authoritative tensor fields to TQ before remote worker dispatch.

        Current rule is intentionally simple and correctness-first:
        - if a required field currently exists in local `batch`, we treat local as authoritative
        - before dispatch, sync those local fields to TQ
        - return KVBatchMeta so the worker path can go through `@tqbridge`

        This gives us the right controller/worker boundary behavior first; more
        selective dirty tracking can be added later.
        """
        if self._kv_meta is None:
            return self

        required_fields = list(required_fields or self._tensor_field_names)
        start_time = time.perf_counter()
        log_dataflow("protocol.lazy.prepare_for_remote.enter", self, required_fields=required_fields)
        local_batch = self._raw_batch()
        if local_batch is not None:
            local_required_fields = [field for field in required_fields if field in local_batch.keys()]
        else:
            local_required_fields = []

        if local_required_fields:
            from roll.utils.transferqueue_utils import kv_batch_meta_put_tensordict

            td = local_batch.select(*local_required_fields)
            self._kv_meta = kv_batch_meta_put_tensordict(
                self._kv_meta,
                td,
                func_name="LazyDataProto.prepare_for_remote",
            )
            self._local_authoritative_fields.difference_update(local_required_fields)
            for field_name in local_required_fields:
                if field_name not in self._tensor_field_names:
                    self._tensor_field_names.append(field_name)

        if hasattr(self._kv_meta, "fields"):
            self._kv_meta.fields = list(dict.fromkeys(self._tensor_field_names))
        if hasattr(self._kv_meta, "extra_info"):
            self._kv_meta.extra_info = {
                "meta_info": copy.deepcopy(self.meta_info),
                "non_tensor_batch": copy.deepcopy(self.non_tensor_batch),
            }
        log_dataflow("protocol.lazy.prepare_for_remote.exit", self._kv_meta, required_fields=required_fields)
        logger.info(
            "[perf][protocol.lazy.prepare_for_remote] "
            f"elapsed={time.perf_counter() - start_time:.6f}s "
            f"required_fields={len(required_fields)} local_required_fields={len(local_required_fields)}"
        )
        return self._kv_meta

    def chunk(self, chunks: int) -> List["DataProto"]:
        if self._kv_meta is None:
            eager_view = DataProto(
                batch=self._raw_batch(),
                non_tensor_batch=self.non_tensor_batch,
                meta_info=self.meta_info,
            )
            return DataProto.chunk(eager_view, chunks)

        from roll.utils.transferqueue_utils import kv_batch_meta2batch_meta, batch_meta2kv_batch_meta

        batch_meta = kv_batch_meta2batch_meta(self._kv_meta)
        sub_batch_metas = batch_meta.chunk(chunks)

        chunks_sizes = [sub_meta.size for sub_meta in sub_batch_metas]
        non_tensor_batch_lst = [{} for _ in range(chunks)]
        for key, val in self.non_tensor_batch.items():
            split_vals = divide_by_chunk_size(val, chunk_sizes=chunks_sizes)
            for idx in range(chunks):
                non_tensor_batch_lst[idx][key] = split_vals[idx]

        local_batch = self._raw_batch()
        overlay_fields = list(self._local_overlay_field_names())
        overlay_batch_lst = None
        if local_batch is not None and overlay_fields:
            overlay_td = local_batch.select(*overlay_fields)
            overlay_batch_lst = divide_by_chunk_size(overlay_td, chunk_sizes=chunks_sizes)

        outputs = []
        for idx, sub_batch_meta in enumerate(sub_batch_metas):
            sub_kv_meta = batch_meta2kv_batch_meta(sub_batch_meta)
            chunked = type(self).from_kv_batch_meta(
                sub_kv_meta,
                tensor_field_names=list(self._tensor_field_names),
                non_tensor_batch=non_tensor_batch_lst[idx],
                meta_info=self.meta_info,
            )
            if overlay_batch_lst is not None:
                object.__setattr__(chunked, "batch", overlay_batch_lst[idx].clone())
                chunked._local_cached_fields.update(
                    set(overlay_batch_lst[idx].keys()) & set(self._local_cached_fields)
                )
                chunked._local_authoritative_fields.update(
                    set(overlay_batch_lst[idx].keys()) & set(self._local_authoritative_fields)
                )
                for field_name in overlay_batch_lst[idx].keys():
                    if field_name not in chunked._tensor_field_names:
                        chunked._tensor_field_names.append(field_name)
                chunked._refresh_materialized_status()
            outputs.append(chunked)
        return outputs

    def select_idxs(self, idxs):
        if self._kv_meta is None:
            return DataProto.select_idxs(self, idxs)

        if isinstance(idxs, list):
            idxs = np.array(idxs)
        elif isinstance(idxs, torch.Tensor):
            idxs = idxs.detach().cpu().numpy()

        if isinstance(idxs, np.ndarray) and idxs.dtype == bool:
            selected_idx = np.nonzero(idxs)[0].tolist()
        elif isinstance(idxs, np.ndarray):
            selected_idx = idxs.tolist()
        else:
            raise TypeError(f"Unsupported idxs type for LazyDataProto.select_idxs: {type(idxs)}")

        kv_meta = copy.deepcopy(self._kv_meta)
        kv_meta.keys = [kv_meta.keys[i] for i in selected_idx]
        if getattr(kv_meta, "tags", None) is not None:
            kv_meta.tags = [kv_meta.tags[i] for i in selected_idx]

        selected_non_tensor = {key: val[selected_idx] for key, val in self.non_tensor_batch.items()}
        result = type(self).from_kv_batch_meta(
            kv_meta,
            tensor_field_names=list(self._tensor_field_names),
            non_tensor_batch=selected_non_tensor,
            meta_info=self.meta_info,
        )
        local_batch = self._raw_batch()
        overlay_fields = self._local_overlay_field_names()
        if local_batch is not None and overlay_fields:
            sliced_tensors = {}
            for key in overlay_fields:
                tensor = local_batch[key]
                if isinstance(tensor, torch.Tensor):
                    indices_t = torch.as_tensor(selected_idx, device=tensor.device)
                    sliced_tensors[key] = tensor[indices_t]
                else:
                    sliced_tensors[key] = tensor[selected_idx]
            if sliced_tensors:
                object.__setattr__(
                    result,
                    "batch",
                    TensorDict(source=sliced_tensors, batch_size=(len(selected_idx),)),
                )
                result._local_cached_fields.update(sliced_tensors.keys())
                result._local_authoritative_fields.update(sliced_tensors.keys())
                for field_name in sliced_tensors.keys():
                    if field_name not in result._tensor_field_names:
                        result._tensor_field_names.append(field_name)
                result._refresh_materialized_status()
        return result

    def slice(self, start=None, end=None, step=None):
        if self._kv_meta is None:
            return DataProto.slice(self, start, end, step)
        indices = np.arange(len(self))[slice(start, end, step)]
        return self.select_idxs(indices)

    def reorder(self, indices):
        if self._kv_meta is None:
            return DataProto.reorder(self, indices)

        indices = indices.view(-1) if isinstance(indices, torch.Tensor) and indices.dim() == 0 else indices
        if isinstance(indices, torch.Tensor):
            indices_np = indices.detach().cpu().numpy()
        else:
            indices_np = np.asarray(indices)
        order = indices_np.tolist()

        self._kv_meta.keys = [self._kv_meta.keys[i] for i in order]
        if getattr(self._kv_meta, "tags", None) is not None:
            self._kv_meta.tags = [self._kv_meta.tags[i] for i in order]
        self.non_tensor_batch = {key: val[indices_np] for key, val in self.non_tensor_batch.items()}
        local_batch = self._raw_batch()
        if local_batch is not None:
            object.__setattr__(self, "batch", local_batch[indices])
            self._refresh_materialized_status()
        return self

    def group_by(self, keys: Union[List[str], str]) -> Dict[str, "DataProto"]:
        if self._kv_meta is None:
            return DataProto.group_by(self, keys)

        keys = self.validate_input(keys)
        assert len(keys) > 0, "Must provide at least one grouping key"
        raw_batch = self._raw_batch()
        missing_keys = [
            key
            for key in keys
            if key not in self.non_tensor_batch and not (raw_batch is not None and key in raw_batch.keys())
        ]
        if missing_keys:
            self.materialize(fields=missing_keys)
            raw_batch = self._raw_batch()

        groups = defaultdict(list)
        for idx in range(len(self)):
            key_values = []
            for key in keys:
                if key in self.non_tensor_batch:
                    key_values.append(str(self.non_tensor_batch[key][idx]))
                else:
                    key_values.append(str(raw_batch[key][idx].detach().cpu().numpy()))
            group_key = "|".join(key_values) if len(key_values) > 1 else key_values[0]
            groups[group_key].append(idx)

        outputs = {group_key: self.select_idxs(indices) for group_key, indices in groups.items()}
        log_dataflow("protocol.lazy.group_by.exit", list(outputs.values()), keys=keys, groups=list(outputs.keys()))
        return outputs

    @staticmethod
    def concat(
        data: List["DataProto"],
        *,
        global_keys: Optional[Set[str]] = None,
    ) -> "DataProto":
        if not data:
            raise ValueError("Cannot concatenate an empty list.")
        log_dataflow("protocol.lazy.concat.enter", data, global_keys=sorted(global_keys or {"metrics"}))

        # Auto-flush: push any locally-authoritative fields to TQ before we
        # test whether a pure-lazy concat is possible.  This lets pipelines
        # write fields (batch["k"] = v) and call concat without ever touching
        # prepare_for_remote() manually.
        for item in data:
            if isinstance(item, LazyDataProto) and item._kv_meta is not None:
                item._flush_authoritative_to_tq()

        all_lazy = all(
            isinstance(item, LazyDataProto) and item._can_use_lazy_meta_ops()
            for item in data
        )
        if not all_lazy:
            eager_data = [
                item.to_dataproto() if isinstance(item, LazyDataProto) else item
                for item in data
            ]
            log_dataflow("protocol.lazy.concat.fallback_eager", eager_data)
            return DataProto.concat(eager_data, global_keys=global_keys)

        from roll.utils.transferqueue_utils import kv_batch_meta2batch_meta, batch_meta2kv_batch_meta

        # batch_meta = type(kv_batch_meta2batch_meta(data[0]._kv_meta)).concat(
        #             [kv_batch_meta2batch_meta(item._kv_meta) for item in data]
        def _concat_safe_batch_meta(kv_meta):
            kv_meta = copy.deepcopy(kv_meta)
            if hasattr(kv_meta, "extra_info"):
                # LazyDataProto already carries authoritative meta_info and
                # non_tensor_batch locally. Stripping extra_info here prevents
                # BatchMeta.concat from treating sample-private generation data
                # as batch-global metadata.
                kv_meta.extra_info = {}
            return kv_batch_meta2batch_meta(kv_meta)

        batch_meta = type(_concat_safe_batch_meta(data[0]._kv_meta)).concat(
            [_concat_safe_batch_meta(item._kv_meta) for item in data]
        )
        merged_kv_meta = batch_meta2kv_batch_meta(batch_meta)

        merged_non_tensor = list_of_dict_to_dict_of_list([item.non_tensor_batch for item in data])
        for key, val in merged_non_tensor.items():
            merged_non_tensor[key] = custom_np_concatenate(val)

        merged_meta = DataProto.concat([DataProto(batch=None, non_tensor_batch={}, meta_info=item.meta_info) for item in data], global_keys=global_keys).meta_info
        tensor_field_names = list(getattr(merged_kv_meta, "fields", None) or getattr(data[0], "_tensor_field_names", []))

        result = LazyDataProto.from_kv_batch_meta(
            merged_kv_meta,
            tensor_field_names=tensor_field_names,
            non_tensor_batch=merged_non_tensor,
            meta_info=merged_meta,
        )
        overlay_field_sets = []
        for item in data:
            if isinstance(item, LazyDataProto):
                overlay_fields = item._local_overlay_field_names()
                if overlay_fields:
                    overlay_field_sets.append(set(overlay_fields))
        if overlay_field_sets:
            common_overlay_fields = set.intersection(*overlay_field_sets)
        else:
            common_overlay_fields = set()

        if common_overlay_fields:
            overlay_tensors = {}
            authoritative_fields = set()
            cached_fields = set()
            for field_name in sorted(common_overlay_fields):
                parts = []
                field_authoritative = False
                field_cached = True
                for item in data:
                    local_batch = item._raw_batch() if isinstance(item, LazyDataProto) else None
                    if local_batch is None or field_name not in local_batch.keys():
                        field_cached = False
                        parts = []
                        break
                    parts.append(local_batch[field_name])
                    if isinstance(item, LazyDataProto) and field_name in item._local_authoritative_fields:
                        field_authoritative = True
                    if not (isinstance(item, LazyDataProto) and field_name in item._local_cached_fields):
                        field_cached = False
                if not parts:
                    continue
                overlay_tensors[field_name] = torch.cat(parts, dim=0)
                if field_authoritative:
                    authoritative_fields.add(field_name)
                if field_cached:
                    cached_fields.add(field_name)

            if overlay_tensors:
                object.__setattr__(
                    result,
                    "batch",
                    TensorDict(source=overlay_tensors, batch_size=(len(result),)),
                )
                result._local_cached_fields.update(cached_fields)
                result._local_authoritative_fields.update(authoritative_fields)
                for field_name in overlay_tensors.keys():
                    if field_name not in result._tensor_field_names:
                        result._tensor_field_names.append(field_name)
                result._refresh_materialized_status()
        log_dataflow("protocol.lazy.concat.exit", result)
        return result


class BatchData:
    """A thin protocol wrapper for dispatch/collect operations.

    This keeps Cluster/decorator code agnostic to whether the payload is a
    DataProto, TensorDict, BatchMeta, or KVBatchMeta.
    """

    def __init__(self, data):
        self._data = data

    @staticmethod
    def _maybe_import_tq_types():
        try:
            from transfer_queue import BatchMeta, KVBatchMeta
            return BatchMeta, KVBatchMeta
        except ImportError:
            return None, None

    def is_chunkable(self) -> bool:
        data = self._data
        BatchMeta, KVBatchMeta = self._maybe_import_tq_types()
        supported = [DataProto, LazyDataProto, TensorDict]
        if BatchMeta is not None:
            supported.extend([BatchMeta, KVBatchMeta])
        return isinstance(data, tuple(supported))

    def is_concatable(self) -> bool:
        data = self._data
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            return False
        sample = data[0]
        BatchMeta, KVBatchMeta = self._maybe_import_tq_types()
        supported = [DataProto, LazyDataProto, ray.ObjectRef, TensorDict, ObjectRefWrap]
        if BatchMeta is not None:
            supported.extend([BatchMeta, KVBatchMeta])
        return isinstance(sample, tuple(supported))

    def chunk(self, chunks: int):
        data = self._data
        BatchMeta, KVBatchMeta = self._maybe_import_tq_types()

        if isinstance(data, (DataProto, LazyDataProto)):
            return data.chunk(chunks=chunks)

        if isinstance(data, TensorDict):
            if len(data) == 0:
                return tuple(TensorDict({}, batch_size=(0,)) for _ in range(chunks))
            index_array = np.arange(len(data))
            chunk_sizes = [len(b) for b in np.array_split(index_array, chunks)]
            raw_chunks = divide_by_chunk_size(data, chunk_sizes=chunk_sizes)
            return tuple(chunk.clone() for chunk in raw_chunks)

        if BatchMeta is not None and isinstance(data, KVBatchMeta):
            from roll.utils.transferqueue_utils import kv_batch_meta2batch_meta

            data = kv_batch_meta2batch_meta(data)

        return data.chunk(chunks=chunks)

    def concat(self, metric_prefix: Optional[str] = None):
        data = self._data
        if not data:
            raise ValueError("Cannot concatenate an empty list of data items.")
        log_dataflow("protocol.batchdata.concat.enter", data)

        sample = data[0]
        BatchMeta, KVBatchMeta = self._maybe_import_tq_types()

        if isinstance(sample, (DataProto, LazyDataProto)):
            result = DataProto.concat(data)
            log_dataflow("protocol.batchdata.concat.exit", result)
            return result

        if isinstance(sample, (ray.ObjectRef, ObjectRefWrap)):
            timeout = None
            if "roll_RPC_TIMEOUT" in os.environ:
                timeout = int(os.environ["roll_RPC_TIMEOUT"])
            total_start = time.perf_counter()

            ray_get_start = time.perf_counter()
            if isinstance(sample, ObjectRefWrap):
                obj_refs = [ref.obj_ref for ref in data]
                fetched = ray.get(obj_refs, timeout=timeout)
                realized = [fetched[i] for i, ref in enumerate(data) if ref.collected]
            else:
                realized = ray.get(data, timeout=timeout)
            ray_get_elapsed = time.perf_counter() - ray_get_start

            if not realized:
                result = DataProto()
                if metric_prefix:
                    result = merge_metrics_into_result(
                        result,
                        {
                            f"time/controller/{metric_prefix}/collect/ray_get": ray_get_elapsed,
                            f"time/controller/{metric_prefix}/collect/concat": 0.0,
                            f"time/controller/{metric_prefix}/collect/total": time.perf_counter() - total_start,
                        },
                    )
                log_dataflow("protocol.batchdata.concat.exit", result)
                return result

            # Re-dispatch based on what workers actually returned.
            concat_start = time.perf_counter()
            result = BatchData(realized).concat()
            concat_elapsed = time.perf_counter() - concat_start
            total_elapsed = time.perf_counter() - total_start
            if metric_prefix:
                result = merge_metrics_into_result(
                    result,
                    {
                        f"time/controller/{metric_prefix}/collect/ray_get": ray_get_elapsed,
                        f"time/controller/{metric_prefix}/collect/concat": concat_elapsed,
                        f"time/controller/{metric_prefix}/collect/total": total_elapsed,
                    },
                )
            logger.info(
                "[perf][protocol.batchdata.concat.objectref] "
                f"prefix={metric_prefix or 'none'} "
                f"ray_get={ray_get_elapsed:.6f}s "
                f"concat={concat_elapsed:.6f}s "
                f"total={total_elapsed:.6f}s "
                f"realized={len(realized)}"
            )
            log_dataflow("protocol.batchdata.concat.exit", result)
            return result

        if isinstance(sample, TensorDict):
            result = torch.cat(data, dim=0)
            log_dataflow("protocol.batchdata.concat.exit", result)
            return result

        if BatchMeta is not None and isinstance(sample, BatchMeta):
            from roll.utils.transferqueue_utils import batch_meta2kv_batch_meta

            def _extract_extra(meta):
                extra = copy.deepcopy(getattr(meta, "extra_info", {}) or {})
                if "meta_info" in extra or "non_tensor_batch" in extra:
                    meta_info = copy.deepcopy(extra.get("meta_info", {}) or {})
                    non_tensor_batch = copy.deepcopy(extra.get("non_tensor_batch", {}) or {})
                    return meta_info, non_tensor_batch
                return extra, {}

            sanitized = []
            meta_only = []
            non_tensor_list = []
            for item in data:
                meta_info, non_tensor_batch = _extract_extra(item)
                sanitized_item = copy.deepcopy(item)
                sanitized_item.extra_info = {}
                sanitized.append(sanitized_item)
                meta_only.append(DataProto(batch=None, non_tensor_batch={}, meta_info=meta_info))
                non_tensor_list.append(non_tensor_batch)

            merged_batch_meta = BatchMeta.concat(sanitized)
            merged_kv_meta = batch_meta2kv_batch_meta(merged_batch_meta)

            merged_non_tensor = list_of_dict_to_dict_of_list(non_tensor_list) if non_tensor_list else {}
            for key, val in merged_non_tensor.items():
                merged_non_tensor[key] = custom_np_concatenate(val)

            merged_meta = DataProto.concat(meta_only).meta_info if meta_only else {}
            tensor_field_names = list(getattr(merged_kv_meta, "fields", None) or [])
            result = LazyDataProto.from_kv_batch_meta(
                merged_kv_meta,
                tensor_field_names=tensor_field_names,
                non_tensor_batch=merged_non_tensor,
                meta_info=merged_meta,
            )
            log_dataflow("protocol.batchdata.concat.exit", result)
            return result

        raise TypeError(f"Unsupported concat sample type: {type(sample)}")

    def prepare_for_remote(self, required_fields: Optional[List[str]] = None):
        data = self._data
        log_dataflow("protocol.batchdata.prepare_for_remote.enter", data, required_fields=required_fields)
        if isinstance(data, LazyDataProto):
            result = data.prepare_for_remote(required_fields=required_fields)
            BatchMeta, KVBatchMeta = self._maybe_import_tq_types()
            if KVBatchMeta is not None and isinstance(result, KVBatchMeta):
                from roll.utils.transferqueue_utils import kv_batch_meta2batch_meta

                result = kv_batch_meta2batch_meta(result)
            log_dataflow("protocol.batchdata.prepare_for_remote.exit", result, required_fields=required_fields)
            return result
        log_dataflow("protocol.batchdata.prepare_for_remote.exit", data, required_fields=required_fields)
        return data


class ObjectRefWrap:
    def __init__(self, obj_ref: ray.ObjectRef, collected=False):
        self.obj_ref = obj_ref
        self.collected = collected
