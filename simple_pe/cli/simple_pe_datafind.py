#! /usr/bin/env python

import os
from argparse import ArgumentParser
from gwpy.timeseries import TimeSeries
import json
import numpy as np
from pesummary.core.command_line import CheckFilesExistAction, DictionaryAction
from . import logger

__author__ = [
    "Charlie Hoy <charlie.hoy@ligo.org>",
]


def command_line():
    """Define the command line arguments for `simple_pe_datafind`
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--outdir",
        help="Directory to store the output",
        default="./",
    )
    parser.add_argument(
        "--channels",
        help=(
            "Channels to use when reading in strain data. Must be provided as "
            "a space separated dictionary with keys giving the ifo and items "
            "giving the channel name, e.g. H1:HWINJ_INJECTED. For GWOSC open "
            "data the dictionary items must be GWOSC, e.g. H1:GWOSC. If you "
            "wish to use simple-pe to produce an injection for you, the "
            "dictionary items must be INJ, e.g. H1:INJ"
        ),
        nargs="+",
        default={},
        action=DictionaryAction,
    )
    parser.add_argument(
        "--injection",
        help=(
            "A json file giving the injection parameters of a signal you "
            "wish to inject"
        ),
        default=None,
    )
    parser.add_argument(
        "--trigger_time",
        help=(
            "Either a GPS time or the event name you wish to analyse. If an "
            "event name is provided, GWOSC is queried to find the event time"
        ),
        default=None,
    )
    return parser


def get_gwosc_data(outdir, event_name, ifo):
    """Fetch GWOSC data with gwpy.timeseries.TimeSeries.fetch_open_data

    Parameters
    ----------
    outdir: str
        directory to output data
    event_name: str
        name of the event you wish to grab the data for
    ifo: str
        name of the IFO you wish to grab data for
    """
    from gwosc.datasets import event_gps
    gps = event_gps(event_name)
    start, stop = int(gps) + 512, int(gps) - 512
    logger.info(
        f"Fetching strain data with: "
        f"TimeSeries.fetch_open_data({ifo}, {start}, {stop})"
    )
    open_data = TimeSeries.fetch_open_data(ifo, start, stop)
    _channel = "GWOSC"
    open_data.name = f"{ifo}:{_channel}"
    open_data.channel = f"{ifo}:{_channel}"
    os.makedirs(f"{outdir}/output", exist_ok=True)
    filename = f"{outdir}/output/{ifo}-{_channel}-{int(gps)}.gwf"
    logger.debug(f"Saving strain data to {filename}")
    open_data.write(filename)
    return filename, _channel


def get_injection_data(outdir, injection, ifo):
    """Create an injection with pycbc and save to a gwf file

    Parameters
    ----------
    outdir: str
        directory to output data
    injection: str
        path to a json file containing the injection parameters
    ifo: str
        name of the IFO you wish to create data for
    """
    # make waveform with independent code: pycbc.waveform.get_td_waveform
    from pycbc.waveform import get_td_waveform, taper_timeseries
    from pycbc.detector import Detector
    with open(injection, "r") as f:
        injection_params = json.load(f)
    # convert to pycbc convention
    params_to_convert = [
        "mass_1", "mass_2", "spin_1x", "spin_1y", "spin_1z",
        "spin_2x", "spin_2y", "spin_2z"
    ]
    logger.info("Generating injection with parameters using pycbc:")
    for param, item in injection_params.items():
        logger.info(f"{param} = {item}")
    for param in params_to_convert:
        injection_params[param.replace("_", "")] = injection_params.pop(param)
    hp, hc = get_td_waveform(**injection_params)
    hp.start_time += injection_params["time"]
    hc.start_time += injection_params["time"]
    ra = injection_params["ra"]
    dec = injection_params["dec"]
    psi = injection_params["psi"]
    ht = Detector(ifo).project_wave(hp, hc, ra, dec, psi)
    ht = taper_timeseries(ht, tapermethod="TAPER_STARTEND")
    prepend = int(512 / ht.delta_t)
    ht.append_zeros(prepend)
    ht.prepend_zeros(prepend)
    strain = TimeSeries.from_pycbc(ht)
    strain = strain.crop(
        injection_params["time"] - 512, injection_params["time"] + 512
    )
    strain.name = f"{ifo}:HWINJ_INJECTED"
    strain.channel = f"{ifo}:HWINJ_INJECTED"
    os.makedirs(f"{outdir}/output", exist_ok=True)
    filename = f"{outdir}/output/{ifo}-INJECTION.gwf"
    logger.debug("Saving injection to {filename}")
    strain.write(filename)
    return filename, "HWINJ_INJECTED"


def get_internal_data(outdir, trigger_time, ifo, channel):
    """Fetch data with gwpy.timeseries.TimeSeries.get

    Parameters
    ----------
    outdir: str
        directory to output data
    trigger_time: float
        central time to grab data for. By default the start time is
        trigger_time-512 and end time if trigger_time+512
    ifo: str
        name of the IFO you wish to grab data for
    channel: str
        name of the channel you wish to grab data for
    """
    gps = float(trigger_time)
    start, stop = int(gps) - 512, int(gps) + 512
    logger.info(
        f"Fetching strain data with: "
        f"TimeSeries.get('{ifo}:{channel}', start={start}, end={stop}, "
        f"verbose=False, allow_tape=True,).astype(dtype=np.float64, "
        f"subok=True, copy=False)"
    )
    data = TimeSeries.get(
        f"{ifo}:{channel}", start=start, end=stop, verbose=False, allow_tape=True,
    ).astype(dtype=np.float64, subok=True, copy=False)
    filename = f"{outdir}/output/{ifo}-{channel}-{int(gps)}.gwf"
    logger.debug(f"Saving strain data to {filename}")
    data.write(filename)
    return filename, channel


def write_cache_file(outdir, strain, channels):
    """Write cache file

    Parameters
    ----------
    outdir: str
        directory to output data
    strain: dict
        dictionary containing the paths to the gwf files for each ifo
    channels: dict
        dictionary containing the channel names for each ifo
    """
    filename = f"{outdir}/output/strain_cache.json"
    logger.info(f"Saving cache file to: {filename}")
    with open(filename, "w") as f:
        _data = {
            ifo: {"strain": strain[ifo], "channel": f"{ifo}:{channels[ifo]}"}
            for ifo in strain.keys()
        }
        json.dump(_data, f)


def main(args=None):
    """Main interface for `simple_pe_datafind`
    """
    parser = command_line()
    opts, _ = parser.parse_known_args(args=args)
    if not os.path.isdir(opts.outdir):
        os.mkdir(opts.outdir)
    _strain, _channels = {}, {}
    for ifo, value in opts.channels.items():
        if "gwosc" in value.lower():
            if opts.trigger_time is None:
                raise ValueError(
                    "Please provide the name of the event you wish to analyse "
                    "via the trigger_time argument"
                )
            _strain[ifo], _channels[ifo] = get_gwosc_data(
                opts.outdir, opts.trigger_time, ifo
            )
        elif "inj" in value.lower():
            if not os.path.isfile(opts.injection):
                raise FileNotFoundError(
                    f"Unable to find file: {opts.injection}"
                )
            _strain[ifo], _channels[ifo] = get_injection_data(
                opts.outdir, opts.injection, ifo
            )
        else:
            if opts.trigger_time is not None:
                _strain[ifo], _channels[ifo] = get_internal_data(
                    opts.outdir, opts.trigger_time, ifo, value
                )
            elif ifo not in strain.keys():
                raise ValueError(f"Please provide a gwf file for {ifo}")
            else:
                raise ValueError("Unable to grab strain data")
    write_cache_file(opts.outdir, _strain, _channels)


if __name__ == "__main__":
    main()
