"""
This module contains utilities for mapping between keys and human-readable
names of variables in the data.
"""

NAME_FOR_KEY = dict({
    "obs": "observation",
    "hid": "hidden state",
    "hmean": "generative mean",
    "hsd": "generative variance",
    "prevhsd": "generative variance",
    "bet": "paddle",
    "betloc": "paddle location",
    "betwid": "paddle width",
    "pe": "prediction error",
    "ape": "absolute prediction error",
    "updtloc": "update in location",
    "updtwid": "update in width",
    "updtmgn": "update magnitude",
    "updtfreq": "update frequency",
    "lr": "apparent learning rate",
    "plrgwid": "p(large paddle)",
    "nextplrgwid": "p(large paddle)",
    "pincwid": "p(increase width)",
    "pdecwid": "p(decrease width)",
    "p_updt_loc": "p(update loc.)",
    "lr_exclude-no-update": "apparent learning rate (excl. no-update)",
    "environment": "environment"
    })
for key, value in list(NAME_FOR_KEY.items()):
    NAME_FOR_KEY[key + "chg"] = "change in" + value
    NAME_FOR_KEY[key + "chgmgn"] = "magnitude of change in" + value
    NAME_FOR_KEY[key + "_bin"] = value + " (binned)"

CPT_NAME_FOR_KEY = dict({key: value.capitalize()
    for key, value in NAME_FOR_KEY.items()})
