"""Plug-in points for segmentation and landmark detection.

The models used in the paper are not included in this release. These abstract
classes define the contract the pipeline expects, so that an independent
implementation can be substituted and the rest of the analysis run unchanged.

Nothing here assumes DICOM, a particular file format, or a particular image
size: a segmenter receives a 2-D array and returns two boolean masks of the
same shape.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class ScarSegmenter(ABC):
    """Produces myocardium and scar masks for one short-axis LGE slice."""

    @abstractmethod
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segment one slice.

        Parameters
        ----------
        image : (H, W) array
            A single short-axis LGE image.

        Returns
        -------
        (myocardium, scar)
            Two boolean arrays of shape (H, W). `scar` must be a subset of
            `myocardium`; any scar pixel outside the myocardium is ignored
            downstream.
        """


class LandmarkDetector(ABC):
    """Locates the LV centre and the RV insertion points on one slice."""

    @abstractmethod
    def detect(self, image: np.ndarray,
               myocardium: np.ndarray) -> Dict[str, Optional[Tuple[float, float]]]:
        """Locate the landmarks for one slice.

        Parameters
        ----------
        image : (H, W) array
            The same image passed to the segmenter.
        myocardium : (H, W) array
            The myocardium mask returned by the segmenter.

        Returns
        -------
        dict
            Must contain:
              'lv_center'             (x, y) centre of the LV cavity
              'rv_insertion_anterior' (x, y) anterior RV insertion point
            May contain:
              'rv_insertion_inferior' (x, y) or None

            Coordinates are in pixels, (x, y), with y increasing downward.
            The anterior insertion point anchors the AHA segment grid, so its
            accuracy determines the segment numbering.
        """


class PrecomputedSegmenter(ScarSegmenter):
    """Returns masks that were supplied up front.

    This is NOT a segmentation model. It exists so the example pipeline can be
    run end to end using the masks shipped in `examples/phantom/`, and so that
    users who already have segmentations can reuse the rest of the pipeline.
    """

    def __init__(self, myocardium: np.ndarray, scar: np.ndarray):
        self._myocardium = np.asarray(myocardium) > 0
        self._scar = np.asarray(scar) > 0

    def segment(self, image: np.ndarray):
        return self._myocardium, self._scar


class PrecomputedLandmarks(LandmarkDetector):
    """Returns landmarks that were supplied up front.

    This is NOT a landmark detection model; see `PrecomputedSegmenter`.
    """

    def __init__(self, lv_center, rv_insertion_anterior,
                 rv_insertion_inferior=None):
        self._landmarks = {
            'lv_center': tuple(lv_center),
            'rv_insertion_anterior': tuple(rv_insertion_anterior),
            'rv_insertion_inferior': (tuple(rv_insertion_inferior)
                                      if rv_insertion_inferior is not None else None),
        }

    def detect(self, image: np.ndarray, myocardium: np.ndarray):
        return dict(self._landmarks)
