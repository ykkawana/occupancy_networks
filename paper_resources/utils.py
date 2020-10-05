
def cutdeci(s, deci=3):
    if isinstance(s, str):
        return s
    deci_str = "{" + ":.{}".format(deci) + "f}"
    return deci_str.format(s)

