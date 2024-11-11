def get_size_in_bits(filters):
    """Total size of all filters in bits."""
    return sum(len(f) for f in filters)

def get_filter_sizes(filters):
    """Get sizes of individual filters in bits."""
    return [len(f) for f in filters]

def count_set_bits_per_filter(filters):
    """Count set bits in each filter."""
    return [sum(1 for bit in filter if bit == '1') for filter in filters]

def get_vector_from_string(bitstring):
    """
    Get a vector representation of the cascade focusing on first three filters.
    
    Args:
        bitstring: Comma-separated string of filter bitstrings
        
    Returns:
        List containing:
        [0] Total size in bits
        [1] Number of filters
        [2] First filter size in bits
        [3] Second filter size in bits (0 if not present)
        [4] Third filter size in bits (0 if not present)
        [5] Set bits in first filter
        [6] Set bits in second filter (0 if not present)
        [7] Set bits in third filter (0 if not present)
    """
    # Parse the string into individual filters
    filters = [f for f in bitstring.split(',') if f]
    
    # Get sizes and set bits for filters
    filter_sizes = get_filter_sizes(filters)
    set_bits = count_set_bits_per_filter(filters)
    
    # Ensure we have values for up to 3 filters, padding with 0s if needed
    sizes = filter_sizes[:3] + [0] * (3 - len(filter_sizes))
    bits = set_bits[:3] + [0] * (3 - len(set_bits))
    
    return [
        float(get_size_in_bits(filters)),
        float(len(filters)),
        float(sizes[0]),
        float(sizes[1]),
        float(sizes[2]),
        float(bits[0]),
        float(bits[1]),
        float(bits[2])
    ]