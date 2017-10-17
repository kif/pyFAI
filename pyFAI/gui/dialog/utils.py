# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""
This module contains utilitaries used by other dialog modules.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/10/2017"

from silx.gui import qt
from silx.gui.hdf5.Hdf5TreeModel import Hdf5TreeModel


def indexFromH5Object(model, h5Object):
    """This code should be inside silx"""
    if h5Object is None:
        return qt.QModelIndex()

    filename = h5Object.file.filename

    # Seach for the right roots
    rootIndices = []
    for index in range(model.rowCount(qt.QModelIndex())):
        index = model.index(index, 0, qt.QModelIndex())
        obj = model.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)
        if obj.file.filename == filename:
            # We can have many roots with different subtree of the same
            # root
            rootIndices.append(index)

    if len(rootIndices) == 0:
        # No root found
        return qt.QModelIndex()

    path = h5Object.name + "/"
    path = path.replace("//", "/")

    # Search for the right node
    found = False
    foundIndices = []
    for _ in range(1000 * len(rootIndices)):
        # Avoid too much iterations, in case of recurssive links
        if len(foundIndices) == 0:
            if len(rootIndices) == 0:
                # Nothing found
                break
            # Start fron a new root
            foundIndices.append(rootIndices.pop(0))

            obj = model.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)
            p = obj.name + "/"
            p = p.replace("//", "/")
            if path == p:
                found = True
                break

        parentIndex = foundIndices[-1]
        for index in range(model.rowCount(parentIndex)):
            index = model.index(index, 0, parentIndex)
            obj = model.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)

            p = obj.name + "/"
            p = p.replace("//", "/")
            if path == p:
                foundIndices.append(index)
                found = True
                break
            elif path.startswith(p):
                foundIndices.append(index)
                break
        else:
            # Nothing found, start again with another root
            foundIndices = []

        if found:
            break

    if found:
        return foundIndices[-1]
    return qt.QModelIndex()


def findClosestSubPath(hdf5Object, path):
    """Find the closest existing path from the hdf5Object using a subset of the
    provided path.

    Returns None if no path found. It is possible if the path is a relative
    path.

    :param h5py.Node hdf5Object: An HDF5 node
    :param str path: A path
    :rtype: str
    """
    if path in ["", "/"]:
        return "/"
    names = path.split("/")
    if path[0] == "/":
        names.pop(0)
    for i in range(len(names)):
        n = len(names) - i
        path2 = "/".join(names[0:n])
        if path2 == "":
            return ""
        if path2 in hdf5Object:
            return path2

    if path[0] == "/":
        return "/"
    return None