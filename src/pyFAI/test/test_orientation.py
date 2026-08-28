#!/usr/bin/env python
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

"""Test suite validating the detector orientation against the position of
control points in the laboratory frame.

The detector `orientation` is only a re-indexing of the pixels: it changes which
array index maps onto a given physical position, it does not move the detector.
Consequently, mirroring the coordinates of a control point and selecting the
matching orientation must give back **exactly** the same position in space.

The reference dataset is synthetic: ~2800 control points spread over 16 LaB6
rings, for the geometry stored in the header of `orientation/orientation_3.edf`.
Reference control points are also provided for the three other orientations,
picked independently (hence a slightly different number of points), which allows
a cross-check that does not depend on the mirroring rule used here.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/08/2026"

import logging
import unittest
from typing import ClassVar

import numpy
from scipy.spatial import cKDTree

from .. import geometry
from ..calibrant import get_calibrant
from ..containers import FixedParameters
from ..control_points import ControlPoints
from ..detectors import detector_factory
from ..geometry.utils import FLIPPED_AXES, convert_orientation
from ..geometryRefinement import GeometryRefinement
from .utilstest import UtilsTest

logger = logging.getLogger(__name__)


class TestOrientationPositions(unittest.TestCase):
    """Position in space of control points across the 4 detector orientations"""

    DETECTOR = "Eiger2_1M"
    WAVELENGTH = 1e-10
    # geometry used to generate the synthetic dataset, see the EDF header
    GEOMETRY: ClassVar[dict] = {"dist": 0.04, "poni1": 0.05, "poni2": 0.06,
                                "rot1": 0.07, "rot2": 0.08, "rot3": 0.0}
    FLIPPED_AXES: ClassVar[dict] = FLIPPED_AXES

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.shape = detector_factory(cls.DETECTOR).shape
        points = numpy.array(cls.load_control_points(3))
        cls.d1 = points[:, 0]
        cls.d2 = points[:, 1]
        cls.ring = points[:, 2].astype(int)

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.d1 = cls.d2 = cls.ring = None

    @staticmethod
    def load_control_points(orientation):
        """Read the reference control points for one orientation

        :param orientation: 1, 2, 3 or 4
        :return: list of [dim1, dim2, ring index]
        """
        filename = UtilsTest.getimage(f"orientation/orientation_{orientation}.npt")
        return ControlPoints(filename).getList()

    def positions_from_file(self, orientation):
        """Position in space of the control points picked for one orientation

        :param orientation: 1, 2, 3 or 4
        :return: (n, 3) array of positions in meter
        """
        points = numpy.array(self.load_control_points(orientation))
        ai = self.build_geometry(orientation)
        return numpy.array(ai.calc_pos_zyx(d1=points[:, 0], d2=points[:, 1])).T

    def build_geometry(self, orientation):
        """Reference geometry, with the detector set to the given orientation

        :param orientation: 1, 2, 3 or 4
        :return: Geometry instance
        """
        detector = detector_factory(self.DETECTOR, {"orientation": orientation})
        return geometry.Geometry(detector=detector, wavelength=self.WAVELENGTH,
                                 **self.GEOMETRY)

    def mirror(self, d1, d2, orientation, center=True):
        """Mirror pixel coordinates to address the same physical position once
        the detector is set to `orientation`.

        Nota: the offset differs with the convention, and both are exact:
        a pixel *index* n has its center at n+0.5, so indexes mirror as
        `shape - 1 - n`, while corner coordinates span [0, shape] and mirror
        as `shape - n`.

        :param d1: coordinates along the slow dimension
        :param d2: coordinates along the fast dimension
        :param orientation: 1, 2, 3 or 4
        :param center: True for pixel centers, False for pixel corners
        :return: 2-tuple of mirrored coordinates
        """
        offset = 1 if center else 0
        flip1, flip2 = self.FLIPPED_AXES[orientation]
        if flip1:
            d1 = self.shape[0] - offset - d1
        if flip2:
            d2 = self.shape[1] - offset - d2
        return d1, d2

    def test_position_is_orientation_invariant(self):
        """The very point of this suite: mirroring the control points and
        selecting the matching orientation gives back the same position in
        space. Would fail if the orientation were applied twice."""
        reference = numpy.array(self.build_geometry(3).calc_pos_zyx(d1=self.d1, d2=self.d2))
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                d1, d2 = self.mirror(self.d1, self.d2, orientation)
                got = numpy.array(self.build_geometry(orientation).calc_pos_zyx(d1=d1, d2=d2))
                delta = abs(got - reference).max()
                self.assertLess(delta, 1e-9,
                                f"orientation {orientation}: position differs by {delta} m")

    def test_detector_position_is_orientation_invariant(self):
        """Same check one layer below, on the detector alone, for both the pixel
        center and the pixel corner conventions."""
        for center in (True, False):
            detector = detector_factory(self.DETECTOR, {"orientation": 3})
            reference = detector.calc_cartesian_positions(self.d1, self.d2, center=center)
            for orientation in (1, 2, 4):
                with self.subTest(orientation=orientation, center=center):
                    d1, d2 = self.mirror(self.d1, self.d2, orientation, center=center)
                    detector = detector_factory(self.DETECTOR, {"orientation": orientation})
                    got = detector.calc_cartesian_positions(d1, d2, center=center)
                    delta = max(abs(a - b).max() for a, b in zip(got[:2], reference[:2]))
                    self.assertLess(delta, 1e-9,
                                    f"orientation {orientation}, center={center}: "
                                    f"position differs by {delta} m")

    def test_cython_matches_python(self):
        """The Cython and the pure numpy implementations agree, whatever the
        orientation."""
        for orientation in (1, 2, 3, 4):
            with self.subTest(orientation=orientation):
                ai = self.build_geometry(orientation)
                cython = numpy.array(ai.calc_pos_zyx(d1=self.d1, d2=self.d2, use_cython=True))
                python = numpy.array(ai.calc_pos_zyx(d1=self.d1, d2=self.d2, use_cython=False))
                delta = abs(cython - python).max()
                self.assertLess(delta, 1e-9, f"orientation {orientation}: "
                                             f"cython and python differ by {delta} m")

    def test_reference_files_describe_the_same_points(self):
        """Strongest cross-check, and it does not use `mirror()` at all: the
        control points picked independently for each orientation must describe
        the same cloud of physical points once each is combined with its own
        orientation.

        The files do not hold exactly the same peaks (they were picked
        separately, hence a slightly different count), so only the bulk of the
        cloud can be compared: unmatched points fall onto a neighbour of the
        same ring, a couple of pixels away. Hence the median."""
        reference = self.positions_from_file(3)
        tree = cKDTree(reference)
        pixel = detector_factory(self.DETECTOR).pixel1
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                distance, _ = tree.query(self.positions_from_file(orientation))
                median = numpy.median(distance) / pixel
                self.assertLess(median, 0.1,
                                f"orientation {orientation}: median distance to the "
                                f"orientation 3 cloud is {median} pixel")

    def test_derived_angles_are_orientation_invariant(self):
        """Since a mirrored control point is the very same point in space, every
        derived quantity must match, `chi` included. This is the user visible
        symptom of applying the orientation twice."""
        ai3 = self.build_geometry(3)
        reference = {"tth": ai3.tth(self.d1, self.d2),
                     "chi": ai3.chi(self.d1, self.d2),
                     "q": ai3.qFunction(self.d1, self.d2)}
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                d1, d2 = self.mirror(self.d1, self.d2, orientation)
                ai = self.build_geometry(orientation)
                for name, expected in (("tth", ai.tth(d1, d2)),
                                       ("chi", ai.chi(d1, d2)),
                                       ("q", ai.qFunction(d1, d2))):
                    delta = abs(expected - reference[name]).max()
                    self.assertLess(delta, 1e-9, f"orientation {orientation}: "
                                                 f"{name} differs by {delta}")

    def test_reference_points_lie_on_rings(self):
        """Cross-check independent of the mirroring rule: the control points
        picked for each orientation land on the expected LaB6 rings."""
        calibrant = get_calibrant("LaB6")
        calibrant.wavelength = self.WAVELENGTH
        expected = numpy.array(calibrant.get_2th())
        for orientation in (1, 2, 3, 4):
            with self.subTest(orientation=orientation):
                points = numpy.array(self.load_control_points(orientation))
                ai = self.build_geometry(orientation)
                tth = ai.tth(points[:, 0], points[:, 1])
                residual = numpy.rad2deg(abs(tth - expected[points[:, 2].astype(int)]))
                self.assertLess(numpy.median(residual), 0.01,
                                f"orientation {orientation}: median residual "
                                f"{numpy.median(residual)} deg")
                self.assertLess(residual.max(), 0.1,
                                f"orientation {orientation}: max residual "
                                f"{residual.max()} deg")


class TestOrientationRefinement(unittest.TestCase):
    """Refining the geometry from the control points must give back the
    reference parameters, whatever the orientation the detector is declared in,
    provided the control points are the ones picked for that orientation.

    `rot3` and the wavelength are fixed during the refinement: the reference
    setup is invariant under `rot3`, which is therefore not constrained by the
    data and is not optimised.
    """

    DETECTOR = TestOrientationPositions.DETECTOR
    WAVELENGTH = TestOrientationPositions.WAVELENGTH
    REFERENCE: ClassVar[dict] = dict(TestOrientationPositions.GEOMETRY)
    # the refinement lands within these of the reference, while a wrong
    # orientation is off by ~2e-2 m and ~1.6e-1 rad, i.e. orders of magnitude more
    TOLERANCE: ClassVar[dict] = {"dist": 1e-4, "poni1": 1e-4, "poni2": 1e-4,
                                 "rot1": 1e-3, "rot2": 1e-3}

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.detector = detector_factory(cls.DETECTOR)
        cls.calibrant = get_calibrant("LaB6")
        cls.calibrant.wavelength = cls.WAVELENGTH
        # refinements are the expensive part, run them once
        cls.matched = {o: cls.refine(cls.control_points(o), o) for o in (1, 2, 3, 4)}
        points3 = cls.control_points(3)
        cls.declared = {o: cls.refine(points3, o) for o in (1, 2, 3, 4)}

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        cls.matched = cls.declared = cls.calibrant = None

    @staticmethod
    def control_points(orientation):
        return numpy.array(TestOrientationPositions.load_control_points(orientation))

    @classmethod
    def refine(cls, points, orientation):
        """Refine the geometry, keeping `rot3` and the wavelength fixed

        :return: dict of refined parameters
        """
        detector = detector_factory(cls.DETECTOR, {"orientation": orientation})
        refiner = GeometryRefinement(data=points, calibrant=cls.calibrant,
                                     detector=detector, wavelength=cls.WAVELENGTH,
                                     dist=0.05,
                                     poni1=detector.shape[0] * detector.pixel1 / 2,
                                     poni2=detector.shape[1] * detector.pixel2 / 2,
                                     rot1=0, rot2=0, rot3=0)
        refiner.refine3(1000000, fix=FixedParameters(["wavelength", "rot3"]))
        return {key: getattr(refiner, key)
                for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3")}

    def test_refinement_is_orientation_invariant(self):
        """Each file refined with its own orientation gives back the reference:
        flipping the image and declaring the matching orientation cancel out."""
        for orientation, found in self.matched.items():
            with self.subTest(orientation=orientation):
                for key, tolerance in self.TOLERANCE.items():
                    self.assertAlmostEqual(found[key], self.REFERENCE[key], delta=tolerance,
                                           msg=f"orientation {orientation}: {key} is "
                                               f"{found[key]} instead of {self.REFERENCE[key]}")

    def test_refinement_differs_with_another_orientation(self):
        """Refining the *same* data while declaring another orientation converges
        just as well but towards different parameters."""
        reference = self.declared[3]
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                found = self.declared[orientation]
                flip1, flip2 = FLIPPED_AXES[orientation]
                if flip1:
                    self.assertGreater(abs(found["poni1"] - reference["poni1"]), 1e-3,
                                       "poni1 should be mirrored")
                    self.assertLess(found["rot2"] * reference["rot2"], 0, "rot2 should change sign")
                if flip2:
                    self.assertGreater(abs(found["poni2"] - reference["poni2"]), 1e-3,
                                       "poni2 should be mirrored")
                    self.assertLess(found["rot1"] * reference["rot1"], 0, "rot1 should change sign")

    def test_convert_orientation_matches_refinement(self):
        """`convert_orientation` reproduces what the refinement finds when the
        same data is declared in another orientation."""
        reference = self.declared[3]
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                expected = convert_orientation(reference, self.detector, 3, orientation)
                found = self.declared[orientation]
                for key, tolerance in self.TOLERANCE.items():
                    self.assertAlmostEqual(found[key], expected[key], delta=tolerance,
                                           msg=f"orientation {orientation}: {key} is "
                                               f"{found[key]}, converted gives {expected[key]}")

    def test_convert_orientation_round_trip(self):
        """Converting back and forth restores the parameters."""
        reference = self.declared[3]
        for orientation in (1, 2, 4):
            with self.subTest(orientation=orientation):
                there = convert_orientation(reference, self.detector, 3, orientation)
                back = convert_orientation(there, self.detector, orientation, 3)
                for key in self.TOLERANCE:
                    self.assertAlmostEqual(back[key], reference[key], places=12,
                                           msg=f"{key} not restored by the round trip")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestOrientationPositions))
    testsuite.addTest(loader(TestOrientationRefinement))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
