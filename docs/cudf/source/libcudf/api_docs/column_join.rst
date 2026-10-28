Column Join
===========

Caching and partitioning joins
------------------------------

Use ``inner_join_match_context`` or ``left_join_match_context`` on a ``hash_join`` object to
determine output memory needs before allocating gather maps. The returned
``join_match_context`` retains match counts and private cached matching groups, so
hash-join retrieval does not repeat the probe. ``sort_merge_join`` returns a
``unique_ptr`` to the same context type for inner joins and retains its left-table
preprocessing state.

.. code-block:: cpp

   auto matches = joiner.inner_join_match_context(left, stream, resources);
   auto const output_bytes = matches.output_size() * 2 * sizeof(cudf::size_type);
   // Reserve output_bytes with the application's memory coordinator, or choose
   // smaller row ranges using matches.match_counts().
   auto maps = joiner.partitioned_inner_join(matches, 0, left.num_rows(), stream, resources);

Retrieval borrows ``matches``. Reuse it with different half-open left-row ranges
without moving it into a partition object. Gather-map indices refer to the
original left table. Counts are in original left-row order; left and full joins
count one placeholder for each unmatched left row. Row partitioning cannot split
one row's matches, so a highly duplicated key may require more than a chosen
output-memory budget.

The context, both input tables, the creating join object, and the resources that
back retained allocations must remain alive until all retrieval operations
complete. Creating the context synchronizes its stream before returning; subsequent
retrieval can use another stream. The memory estimate above covers the two
gather maps, excluding allocator overhead, retained match state, and scratch.
For a nonempty hash-join right table, the retained device match state uses two
``size_type`` values per left row (one count and one group index), or 8 bytes per
row before allocator rounding. Partition retrieval also needs temporary offsets.

``hash_join::full_join_match_context`` prepares the left-side portion of a full join.
After retrieving the partitions, use ``finalize_partitioned_full_join`` to add
unmatched right rows. ``output_size()`` excludes those final unmatched rows.

Migration to accessors
----------------------

The existing ``join_match_context``, ``join_partition_context``, and
``*_join_match_context()`` factory names and return types remain available. Existing
partition-context calls also reuse the cached hash probe state. Overloads accepting
an explicit ``cudf::memory_resources`` bundle allow output and scratch allocations
to use separate resources.

Direct access to ``_left_table`` and ``_match_counts`` is deprecated in 26.12. These
public fields will be removed in 27.02. Replace those accesses with ``left_table()``
and ``match_counts()``. The current pointer storage for counts remains during this
transition; removing that indirection is deferred until the public field is removed.
The context and factory names are not deprecated.

Both input data and the context's match data must remain unchanged while reusing the
cache. Contexts constructed directly from external counts retain uncached hash
retrieval, and do not provide a cached ``output_size()``. A context returned by a
join factory must be retrieved using that join object and the matching join kind.

.. doxygengroup:: column_join
   :members:
