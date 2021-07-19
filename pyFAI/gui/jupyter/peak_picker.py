#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2021 European Synchrotron Radiation Facility, Grenoble, France
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""Semi-graphical tool for peak-picking and extracting visually control points
from an image with Debye-Scherer rings in Jupyter environment"""

__authors__ = ["Philipp Hans", "Jérôme Kieffer"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/07/2021"
__status__ = "production"

from matplotlib.pyplot import subplots
import itertools
import ipywidgets as widgets
from ..peak_picker import PeakPicker as _PeakPicker, preprocess_image

class PeakPicker(_PeakPicker):
    """
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ics = {}
        self.image = self.data
        self.figsize = (10, 8)
        self.current_points = None    
        self.reset_picked()
    
    def reset_picked(self):
        self.dict_annotated = {} 
        self.dict_all_points = {}
    
    def merge_segments(self):
        work = self.dict_all_points
        for key in work.keys():
            sets = work[key]
            work[key] = [list(itertools.chain.from_iterable(sets))]
    
    def add_to_selected_ring(self, ring, current_points):
        """
        ring is integer from
        first ring has index 0
        """
        work = self.dict_all_points
        key = ring
        if work.get(key) is None:
            work[key] = [current_points]
        else:
            work[key].append(current_points)
    
    def get_ring(self, ring):
        """
        maybe delete
        """
        self.current_ring = ring
    
    def gui(self, log=False, maximize=False, pick=True):
        """
        :param log: show z in log scale
        """
        ### Several nested functions:
        def merge_segments_on_click(*kargs, **kwargs):
            self.merge_segments()

        def add_to_ring_on_click(*kargs, **kwargs):
            self.add_to_selected_ring(ringer.value, self.current_points)
        
        def reset_picking_on_click(*kargs, **kwargs):
            self.reset_picked()

        def save_ctrl_pts_on_click(b):
            filename = text_field_output_name.value
            work = self.dict_all_points
            for key in work.keys():
                self.points.append(work[key][0], key)
            self.points.save(filename)

        print('ATTENTION!!!!!!')
        print('you must click add to selected ring after eack pick')
        print('pick with shift and left mouse button')
        print('then you click merge')
        print('then click save')
        
        
        self.fig, self.ax = subplots(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111)

        ringer = widgets.IntText(description="Ring#", continuous_update=True)

        button_merge = widgets.Button(description='merge rings')
        button_merge.on_click(merge_segments_on_click)
        
        button_add = widgets.Button(description='add_to_selected_ring')
        button_add.on_click(add_to_ring_on_click)
        
        button_reset = widgets.Button(description='reset')
        button_reset.on_click(reset_picking_on_click)
        
        text_field_output_name = widgets.Text(
            value='./controlpoint_file.npt',
            #placeholder='Type something',
            description='control-point file:',
            disabled=False
        )        
        button_save_ctrl_pts = widgets.Button(description='save control-points file')
        button_save_ctrl_pts.on_click(save_ctrl_pts_on_click)
        
        _ = widgets.display(widgets.VBox([
                                widgets.HBox([ringer, button_add, button_merge, button_reset]), 
                                widgets.HBox([text_field_output_name, button_save_ctrl_pts])]))
                
        disp_image, bounds = preprocess_image(self.data, log)
        show_min, show_max = bounds 
        self.im = self.ax.imshow(disp_image, 
                                  cmap='magma', 
                                  origin="lower",
                                  vmin=show_min, 
                                  vmax=show_max,
                                  )
        self.ax.set_ylabel('y in pixels')
        self.ax.set_xlabel('x in pixels')
        self.ax.images[0].set_picker(True)
        self.cid = self.fig.canvas.mpl_connect("pick_event", self.pick)

    def pick(self, event):
        e = event.mouseevent
        if (e.inaxes and e.button == 1 and e.key == 'shift'):
            x, y = int(e.xdata), int(e.ydata)
            key = (x, y)
            if key in self.dict_annotated:
                annot = self.dict_annotated[key]
                annot.set_visible(False)
                annot.remove()
                del self.dict_annotated[key]
            else:
                pt0x, pt0y = key
                """data"""
                
                annot = self.ax.annotate("", xy=(1, 1), xytext=(-1, 1),
                                     textcoords="offset points",
                                     arrowprops=dict(arrowstyle="->", color="w",
                                                     connectionstyle="arc3"),
                                     va="bottom", ha="left", fontsize=10,
                                     bbox=dict(boxstyle="round", fc="r"))
                annot.set_visible(False)
                self.dict_annotated[key] = annot
            self.annotate(key)
            ypix, xpix = int(e.ydata + 0.5), int(e.xdata + 0.5)
            try:
                points = self.massif.find_peaks([ypix, xpix],
                                            self.defaultNbPoints,
                                            None, None)
                self.current_points = points
            except Exception as e:
                points = []
            self.annotate_found(points)
    
    def annotate_found(self, points):
        for X in points:
            key = [int(x) for x in X]
            
            y, x = key
            annot = self.ax.annotate("", xy=(x, y), xytext=(-1, 1),
                                 textcoords="offset points",
                                 arrowprops=dict(arrowstyle="->", color="w",
                                                 connectionstyle="arc3"),
                                 va="bottom", ha="left", fontsize=10,
                                 bbox=dict(boxstyle="round", fc="r"))
            b3 = widgets.Button(description='button 3')
            annot.set_text('•')
            annot.xy = (x, y)
            annot.set_visible(True)
            self.dict_annotated[key] = annot
            self.fig.canvas.draw_idle()        
        
    def annotate(self, X):
        key = X
        if key in self.dict_annotated:
            annot = self.dict_annotated.get(key)
            annot.set_visible(True)
            annot.set_text('•')
            annot.xy = key
        print(self.dict_annotated)

        self.fig.canvas.draw_idle()