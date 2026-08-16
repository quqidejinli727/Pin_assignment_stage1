"""Read the tensor subset used by FloorSet ``.pth`` files without PyTorch.

FloorSet's public Lite test files use PyTorch's zip-based serialization and
contain plain CPU tensors.  Installing PyTorch solely to decode these files is
unnecessarily heavy for the converter, so this module implements a restricted
reader for that exact subset.  Unknown pickle globals and non-CPU storages are
rejected.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


_DTYPES = {
    "BoolStorage": np.dtype("?"),
    "ByteStorage": np.dtype("u1"),
    "CharStorage": np.dtype("i1"),
    "ShortStorage": np.dtype("<i2"),
    "IntStorage": np.dtype("<i4"),
    "LongStorage": np.dtype("<i8"),
    "HalfStorage": np.dtype("<f2"),
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
}


@dataclass(frozen=True)
class _StorageType:
    name: str


@dataclass(frozen=True)
class _Storage:
    values: np.ndarray


def _rebuild_tensor(
    storage: _Storage,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    *_: Any,
) -> np.ndarray:
    """Rebuild a CPU tensor view as an independent NumPy array."""
    shape = tuple(int(value) for value in size)
    element_strides = tuple(int(value) for value in stride)
    offset = int(storage_offset)
    if offset < 0:
        raise ValueError(f"Negative tensor storage offset: {offset}")
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"Negative tensor dimension: {shape}")
    if any(value < 0 for value in element_strides):
        raise ValueError(f"Negative tensor stride is unsupported: {element_strides}")
    if not shape:
        if offset >= storage.values.size:
            raise ValueError("Scalar tensor offset exceeds its storage.")
        return np.asarray(storage.values[offset]).copy()
    max_index = offset + sum(
        (dimension - 1) * stride_value
        for dimension, stride_value in zip(shape, element_strides)
        if dimension
    )
    if max_index >= storage.values.size:
        raise ValueError(
            f"Tensor view exceeds storage: max_index={max_index}, "
            f"storage={storage.values.size}."
        )
    base = storage.values[offset:]
    byte_strides = tuple(value * storage.values.dtype.itemsize for value in element_strides)
    view = np.lib.stride_tricks.as_strided(base, shape=shape, strides=byte_strides)
    return np.array(view, copy=True)


def _rebuild_parameter(data: np.ndarray, *_: Any) -> np.ndarray:
    return data


class _RestrictedTorchUnpickler(pickle.Unpickler):
    """Unpickler allowing only the globals required by plain CPU tensors."""

    def __init__(self, stream: io.BytesIO, storage_loader: Callable[[tuple[Any, ...]], _Storage]):
        super().__init__(stream)
        self._storage_loader = storage_loader

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        if module == "torch._utils" and name in {"_rebuild_tensor", "_rebuild_tensor_v2"}:
            return _rebuild_tensor
        if module == "torch._utils" and name in {"_rebuild_parameter", "_rebuild_parameter_with_state"}:
            return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")

    def persistent_load(self, persistent_id: Any) -> _Storage:
        if not isinstance(persistent_id, tuple):
            raise pickle.UnpicklingError("Unsupported non-tuple persistent id.")
        return self._storage_loader(persistent_id)


def load_torch_zip(path: str | Path) -> Any:
    """Load a FloorSet tensor archive into lists and NumPy arrays.

    This function intentionally supports only CPU tensor storages.  It does not
    execute arbitrary Python globals embedded in the pickle.
    """
    source = Path(path)
    with zipfile.ZipFile(source) as archive:
        pickle_names = [name for name in archive.namelist() if name.endswith("data.pkl")]
        if len(pickle_names) != 1:
            raise ValueError(f"Expected one data.pkl in {source}, found {pickle_names!r}.")
        pickle_name = pickle_names[0]
        prefix = pickle_name[: -len("data.pkl")]
        storage_cache: dict[str, _Storage] = {}

        def load_storage(persistent_id: tuple[Any, ...]) -> _Storage:
            if len(persistent_id) < 5 or persistent_id[0] != "storage":
                raise pickle.UnpicklingError(f"Unsupported persistent id: {persistent_id!r}")
            storage_type, key, location, element_count = persistent_id[1:5]
            if not isinstance(storage_type, _StorageType) or storage_type.name not in _DTYPES:
                raise pickle.UnpicklingError(f"Unsupported storage type: {storage_type!r}")
            if str(location) != "cpu":
                raise pickle.UnpicklingError(f"Only CPU storages are supported, got {location!r}.")
            cache_key = str(key)
            if cache_key not in storage_cache:
                raw = archive.read(f"{prefix}data/{cache_key}")
                dtype = _DTYPES[storage_type.name]
                values = np.frombuffer(raw, dtype=dtype)
                expected = int(element_count)
                if values.size < expected:
                    raise ValueError(
                        f"Storage {cache_key} is truncated: {values.size} < {expected}."
                    )
                storage_cache[cache_key] = _Storage(values[:expected])
            return storage_cache[cache_key]

        payload = archive.read(pickle_name)
        return _RestrictedTorchUnpickler(io.BytesIO(payload), load_storage).load()
