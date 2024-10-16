# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.strings.convert.convert_floats cimport (
    is_float as cpp_is_float,
)

from cudf._lib.column cimport Column


@acquire_spill_lock()
def is_float(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that have floats.
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_strings.view()

    with nogil:
        c_result = move(cpp_is_float(
            source_view
        ))

    return Column.from_unique_ptr(move(c_result))
