# !/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026-2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Helper functions around the geometry, mostly conversions between the
different detector orientation conventions.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/08/2026"
__status__ = "production"
__docformat__ = 'restructuredtext'

__all__ = ["FLIPPED_AXES", "convert_orientation"]

FLIPPED_AXES = {1: (True, True),
                2: (True, False),
                3: (False, False),
                4: (False, True)}
"""Which axis is mirrored with respect to the native orientation (3),
as a 2-tuple (slow/dim1, fast/dim2)."""


def convert_orientation(parameters, detector, from_orientation, to_orientation):
    """Re-express a geometry so that the *same pixel data* is described with
    another detector orientation.

    Mirroring an axis moves the PONI to the opposite side of the detector and
    reverses the sense of the rotation about the *other* in-plane axis:

    * slow axis mirrored: ``poni1 -> L1 - poni1`` and ``rot2 -> -rot2``
    * fast axis mirrored: ``poni2 -> L2 - poni2`` and ``rot1 -> -rot1``

    where ``L1`` and ``L2`` are the detector sizes along the slow and the fast
    dimension. ``dist`` and the wavelength are invariant. ``rot3`` is left
    untouched: a setup which is invariant under ``rot3`` does not constrain it.

    Typical use: reading a calibration performed by a program which flips the
    image, such as *Dioptas* (orientation 2), to process the unflipped data.

    Nota: this converts a geometry meant to describe *unchanged* data. When the
    image itself is flipped as well, the two effects cancel out and the
    parameters are unchanged.

    :param parameters: dict with at least dist, poni1, poni2, rot1 and rot2
    :param detector: detector instance, used for its shape and pixel sizes
    :param from_orientation: orientation the parameters are expressed in, 1 to 4
    :param to_orientation: wanted orientation, 1 to 4
    :return: new dict of parameters, the input is left untouched
    """
    for orientation in (from_orientation, to_orientation):
        if orientation not in FLIPPED_AXES:
            raise ValueError(f"Unsupported orientation {orientation}, expected 1 to 4")
    converted = dict(parameters)
    flip1 = FLIPPED_AXES[from_orientation][0] != FLIPPED_AXES[to_orientation][0]
    flip2 = FLIPPED_AXES[from_orientation][1] != FLIPPED_AXES[to_orientation][1]
    if flip1:
        converted["poni1"] = detector.shape[0] * detector.pixel1 - parameters["poni1"]
        converted["rot2"] = -parameters["rot2"]
    if flip2:
        converted["poni2"] = detector.shape[1] * detector.pixel2 - parameters["poni2"]
        converted["rot1"] = -parameters["rot1"]
    return converted
