# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2024-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""This modules contains helper function to convert to/from crystfel format described in:

https://gitlab.desy.de/thomas.white/crystfel/-/blob/master/doc/man/crystfel_geometry.5.md

Luckily (xyz) matches pyFAI's definition...
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/08/2024"
__status__ = "production"

import logging
import numpy
from ..detectors import detector_factory
from ..units import hc
from .. import load
logger = logging.getLogger(__name__)


def parse_crystfel_geom(fname):
    """Generic parser for CrystFEL geometry files

    :param fname: filename of the geometry file
    :return: dict with the parsed structure.
    """
    res = {}
    with open(fname, "r") as f:
        for line in f:
            eol = line.find(";")-1
            if eol>0:
                line = line[:eol]
            if "=" in line:
                lhs, rhs = [i.strip() for i in line.split("=", 1)]
                try:
                    rhs = float(rhs)
                except:
                    pass
                if "/" in lhs:
                    grp,val = [i.strip() for i in lhs.split("/", 1)]
                    if grp not in res:
                        res[grp] = {}
                    res[grp][val] = rhs
                else:
                    res[lhs] = rhs
    return res

def build_detector(config):
    """Build a detector from the parsed config

    :param config: dict as parsed by parse_crystfel_geom
    :return: Detector instance
    """
    max_ss = 0
    max_fs = 0
    for module in config.values():
        if isinstance(module, dict):
            max_ss = max(max_ss, int(module.get("max_ss", 0)))
            max_fs = max(max_fs, int(module.get("max_fs", 0)))
    pixel_size = 1.0/config["res"]
    detector = detector_factory("Detector", {"pixel1": pixel_size,
                                            "pixel2": pixel_size,
                                            "max_shape":(max_ss+1, max_fs+1),
                                            "orientation":3})
    mask = numpy.zeros(detector.shape, numpy.int8)
    for name, module in config.items():
        if isinstance(module, dict) and name.startswith("bad"):
            if "min_fs" in module and \
               "min_ss" in module and \
               "max_fs" in module and \
               "max_ss" in module:
                fs_slice = slice(int(module["min_fs"]), int(module["max_fs"])+1)
                ss_slice = slice(int(module["min_ss"]), int(module["max_ss"])+1)
                mask[ss_slice, fs_slice] = 1
    detector.mask = mask
    return detector



def build_geometry(config):
    """Build a detector from the parsed config

    :param config: dict as parsed by parse_crystfel_geom
    :return: Detector instance
    """
    detector = build_detector(config)
    x = numpy.zeros(detector.shape)
    y = numpy.zeros(detector.shape)
    mask = numpy.ones(detector.shape, numpy.int8)

    for name,module in config.items():
        if isinstance(module, dict) and not (name.startswith("bad") or
                                            name.startswith("group") or
                                            name.startswith("rigid_group")):
            if ("corner_x" in module and
                "corner_y" in module and
                "fs"  in module and
                "ss"  in module and
                "min_ss" in module and
                "min_fs" in module and
                "max_ss" in module and
                "max_fs" in module):

                sl = (slice(int(module["min_ss"]), int(module["max_ss"]+1)),
                      slice(int(module["min_fs"]), int(module["max_fs"]+1)))
                fs = numpy.outer(numpy.ones(sl[0].stop-sl[0].start),
                                 numpy.arange(sl[1].stop-sl[1].start))
                ss = numpy.outer(numpy.arange(sl[0].stop-sl[0].start),
                                 numpy.ones(sl[1].stop-sl[1].start))

                x[sl] += module["corner_x"]
                y[sl] += module["corner_y"]

                f = {i[-1]:float(i[:-1]) for i in module["fs"].split()}
                s = {i[-1]:float(i[:-1]) for i in module["ss"].split()}

                rotatation = numpy.array([[f["x"], f["y"]],
                                          [s["x"], s["y"]]])
                inv_rotation = numpy.linalg.inv(rotatation)

                x[sl] += inv_rotation[0,0] * fs + inv_rotation[0,1] * ss
                y[sl] += inv_rotation[1,0] * fs + inv_rotation[1,1] * ss
                mask[sl] = 0
    xmin = x.min()
    ymin = y.min()
    detector.mask = numpy.logical_or(detector.mask, mask)

    x-=xmin
    y-=ymin

    pos = numpy.zeros(detector.shape+(4,3))
    pos[:,:, 0, 1] = (y - 0.5) * detector.pixel1
    pos[:,:, 0, 2] = (x - 0.5) * detector.pixel2
    pos[:,:, 1, 1] = (y + 0.5) * detector.pixel1
    pos[:,:, 1, 2] = (x - 0.5) * detector.pixel2
    pos[:,:, 2, 1] = (y + 0.5) * detector.pixel1
    pos[:,:, 2, 2] = (x + 0.5) * detector.pixel2
    pos[:,:, 3, 1] = (y - 0.5) * detector.pixel1
    pos[:,:, 3, 2] = (x + 0.5) * detector.pixel2
    detector.set_pixel_corners(pos)

    # manage energy/wavelength:
    wavelength = None
    if "photon_energy" in config:
        energy_split = config["photon_energy"].split()
        if len(energy_split) >= 2:
            energy_val, energy_unit = energy_split[:2]
            pref = 1e-7
            if energy_unit=="eV":
                pref = 1e-7
            elif energy_unit=="keV":
                pref = 1e-10
            else:
                logger.warning("Unknown energy unit %s", energy_unit)
        elif len(energy_split) == 1:
            pref = 1e-7
        wavelength = pref*hc/float(energy_val)
    elif "wavelength" in config:
        wl_split = config["wavelength"].split()
        if len(energy_split) >= 2:
            wl_val, wl_unit = wl_split[:2]
            if wl_unit=="A":
                    pref = 1e-10
            elif wl_unit=="m":
                    pref = 1.0
        else:
            pref = 1
        wavelength = pref*float(wl_val)

    return load({"detector": detector,
                 "dist": config.get("clen", 1),
                 "poni1":-ymin*detector.pixel1,
                 "poni2":-xmin*detector.pixel1,
                 "wavelength":wavelength})
