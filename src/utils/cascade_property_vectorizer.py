import base64


def get_size_in_bits(filters):
    return sum(len(f) for f in filters)


def get_filter_sizes(filters):
    return [len(f) for f in filters]


def count_set_bits_per_filter(filters):
    return [sum(1 for bit in filter if bit == "1") for filter in filters]


def compact_string_to_bitstring(compact_string):
    cascade = base64.urlsafe_b64decode(compact_string)
    return "".join(format(byte, "08b") for byte in cascade)


def get_vector_from_string(bitstring):
    # Parse the string into individual filters
    filters = [compact_string_to_bitstring(f) for f in bitstring.split(",") if f]

    # Get sizes and set bits for filters
    filter_sizes = get_filter_sizes(filters[:3])
    set_bits = count_set_bits_per_filter(filters[:3])

    # Ensure we have values for up to 3 filters, padding with 0s if needed
    sizes = filter_sizes + [0] * (3 - len(filter_sizes))
    bits = set_bits + [0] * (3 - len(set_bits))

    #assert sizes[0] == len(filters[0]) * 6

    return [
        float(get_size_in_bits(filters)),
        float(len(filters)),
        float(sizes[0]),
        float(sizes[1]),
        float(sizes[2]),
        float(bits[0]),
        float(bits[1]),
        float(bits[2]),
        float(sizes[0] / sizes[1]),
    ]
