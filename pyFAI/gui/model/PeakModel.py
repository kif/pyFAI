# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
#
# ###########################################################################*/

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "21/03/2019"

import numpy
from .AbstractModel import AbstractModel


class PeakModel(AbstractModel):

    def __init__(self, parent=None):
        super(PeakModel, self).__init__(parent)
        self.__name = None
        self.__color = None
        self.__coords = []
        self.__ringNumber = None
        self.__isEnabled = True
        self.__numpyCoords = None

    def __len__(self):
        return len(self.__coords)

    def isValid(self):
        return self.__name is not None and self.__ringNumber is not None

    def name(self):
        return self.__name

    def setName(self, name):
        self.__name = name
        self.wasChanged()

    def isEnabled(self):
        """
        True if this group have to be taken into acount.

        :rtype: bool
        """
        return self.__isEnabled

    def setEnabled(self, isEnabled):
        """
        Set if this group have to be taken into acount.

        :param bool isEnabled: True to enable this group.
        """
        if self.__isEnabled == isEnabled:
            return
        self.__isEnabled = isEnabled
        self.wasChanged()

    def color(self):
        return self.__color

    def setColor(self, color):
        self.__color = color
        self.wasChanged()

    def coords(self):
        return self.__coords

    def setCoords(self, coords):
        self.__coords = coords
        self.__numpyCoords = None
        self.wasChanged()

    def mergeCoords(self, coords):
        """Merge new coords to the current list of coords.

        Duplicated values are removed from the new coords, and the is added
        the end of the previous list.
        """
        new_coords = set(coords) - set(self.__coords)
        self.__coords += list(new_coords)
        self.__numpyCoords = None
        self.wasChanged()

    def ringNumber(self):
        return self.__ringNumber

    def setRingNumber(self, ringNumber):
        assert(ringNumber >= 1)
        self.__ringNumber = ringNumber
        self.wasChanged()

    def copy(self, parent=None):
        peakModel = PeakModel(parent)
        peakModel.setName(self.name())
        peakModel.setColor(self.color())
        peakModel.setCoords(list(self.coords()))
        peakModel.setRingNumber(self.ringNumber())
        peakModel.setEnabled(self.isEnabled())
        return peakModel

    def distanceTo(self, coord):
        """Returns the smallest distance to this coord.

        None is retruned if the group contains no peaks.

        :param Tuple[float,float] coord: Distance to mesure
        """
        if len(self.__coords) == 0:
            return None
        if self.__numpyCoords is None:
            self.__numpyCoords = numpy.array(self.__coords)
        coord = numpy.array(coord)
        distances = numpy.linalg.norm(self.__numpyCoords - coord, axis=1)
        return distances.min()
