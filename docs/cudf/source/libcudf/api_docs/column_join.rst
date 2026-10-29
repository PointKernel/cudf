Column Join
===========

Caching and partitioning joins
------------------------------

Use ``inner_join_match_context`` or ``left_join_match_context`` to compute match
counts and output size before allocating gather maps. Hash joins cache matching
groups to avoid probing again; sort-merge joins retain left-table preprocessing.

.. code-block:: cpp

   auto matches = joiner.inner_join_match_context(left, stream, resources);
   auto const output_bytes = matches.output_size() * 2 * sizeof(cudf::size_type);
   // Reserve output_bytes before allocating the gather maps.
   auto maps = joiner.partitioned_inner_join(matches, 0, left.num_rows(), stream, resources);

This example uses ``hash_join``. ``sort_merge_join::inner_join_match_context``
returns a ``unique_ptr<join_match_context>``; pass ``*matches`` to retrieval.
The context can be reused with different half-open left-row ranges. Left indices
refer to the original table. Use ``match_counts()`` to choose partition bounds;
a partition cannot split one row's matches. See ``join_match_context`` for
lifetime and stream requirements.

``output_bytes`` covers the two gather maps, excluding retained match state,
temporary storage, and allocator overhead. For a nonempty hash-join build table,
retained device state uses two ``size_type`` values per left row (8 bytes).
Retrieval also allocates temporary offsets.

``full_join_match_context`` counts the left-side output only. After retrieving
all partitions, call ``finalize_partitioned_full_join`` to add unmatched right rows.

Migration to accessors
----------------------

Direct access to ``_left_table`` and ``_match_counts`` is deprecated in 26.12 and
will be removed in 27.02. Use ``left_table()`` and ``match_counts()`` instead.
Existing context types, factory names, and partition-context calls remain available.

.. doxygengroup:: column_join
   :members:
