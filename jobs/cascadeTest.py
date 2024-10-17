import cascadeUtils


def run(params):
    r = params["r"]
    s = params["s"]
    parallelize = params["parallelize"]
    rset = cascadeUtils.generate_id_set(r)
    sset = cascadeUtils.generate_id_set(s)

    cascade = cascadeUtils.create_padded_cascade(
        rset,
        sset,
        len(rset),
        len(sset),
        multi_process=parallelize,
    )

    for elem in rset:
        if elem not in cascade:
            return {"message": "Failed to confirm expected member"}
    for elem in sset:
        if elem in cascade:
            return {"message": "Failed to confirm expected exclusion"}

    return {"message": "Cascade works"}
