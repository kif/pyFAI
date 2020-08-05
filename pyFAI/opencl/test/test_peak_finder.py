#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Project: PyFAI: diffraction signal analysis
#             https://github.com/silx-kit/pyFAI
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

"""
Simple test of peak-pickers within pyFAI
"""

__authors__ = ["Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "2020 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/08/2020"

import logging
import numpy

import unittest
from .. import ocl
if ocl:
    import pyopencl.array
import fabio
from ...test.utilstest import UtilsTest
from ...azimuthalIntegrator import AzimuthalIntegrator
from ..peak_finder import OCL_SimplePeakFinder
logger = logging.getLogger(__name__)


@unittest.skipIf(UtilsTest.opencl is False, "User request to skip OpenCL tests")
@unittest.skipUnless(ocl, "PyOpenCl is missing")
class TestOclPeakFinder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestOclPeakFinder, cls).setUpClass()
        if ocl:
            if logger.getEffectiveLevel() <= logging.INFO:
                cls.PROFILE = True
            else:
                cls.PROFILE = False
        cls.ref = numpy.array([(  88,  705), (1097,  907), ( 833,  930), (1520, 1083),
                               (1463, 1249), (1721, 1281), (1274, 1316), (1662, 1372),
                               ( 165, 1433), ( 304, 1423), (1058, 1449), (1260, 1839),
                               ( 806, 2006), ( 129, 2149), (1981, 2272), (1045, 2446)],
                               dtype=[('x', '<i4'), ('y', '<i4')])
        cls.img = fabio.open(UtilsTest.getimage("Pilatus6M.cbf")).data
        cls.ai = AzimuthalIntegrator.sload(UtilsTest.getimage("Pilatus6M.poni"))

    @classmethod
    def tearDownClass(cls):
        super(TestOclPeakFinder, cls).tearDownClass()
        cls.ai = None
        cls.img = None
        cls.ref = None

    @unittest.skipUnless(ocl, "pyopencl is missing")
    def test_simple_peak_finder(self):
        """
        test for simple peak picker
        """
        
        msk = self.img<0
        pf = OCL_SimplePeakFinder(mask=msk)
        res = pf(self.img, window=11)
        s1 = set((i["x"], i["y"]) for i in self.ref) 
        s2 = set((i["x"], i["y"]) for i in res)
        self.assertGreater(len(res), len(self.ref), "Many more peaks with default settings")            
        self.assertFalse(bool(s1.difference(s1.intersection(s2))), "All peaks found")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testSuite = unittest.TestSuite()
    testSuite.addTest(loader(TestOclPeakFinder))
    return testSuite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")