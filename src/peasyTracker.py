"""
This script is a conversion of SimpleTracker Matlab Addon created by Jean-Yves Tinevez to Python code using opensource libraries.

Date: 25/06/2024
Author: Jacques Battaglia, jacques.battaglia@cnrs.fr; Jean-Baptiste Deloges, jean-baptiste.deloges@sorbonne-universite.fr
"""

# Libraries
import itertools
import numpy
from scipy import sparse, optimize, spatial
from typing import Literal


def HungarianLinker(
    src: numpy.ndarray[Literal["nSrcPts", "nDims"], float],
    tgt: numpy.ndarray[Literal["nTgtPts", "nDims"], float],
    d_max: float = numpy.inf,
) -> tuple[numpy.ndarray[Literal["nMatchedPts"], int], numpy.ndarray[Literal["nMatchedPts"], int]]:
    """
    HungarianLinker paires a point of src and tgt variables while minimizing the overall sum of distances between the points of a pair. If a distance exceeds d_max, then it is set to the maximum of the distance matrix so that the associated points cannot be matched.

    Args:
        * src, numpy.ndarray[[nSrcPts, nDims], float]: Point matrix for a frame
        * tgt, numpy.ndarray[[nTgtPts, nDims], float]: Point matrix for another frame
        * d_max, float: Distance threshold at which points are not paired

    Returns:
        matchedSrc, numpy.ndarray[[nMatchedPts,], int]: Indices for matched points in the range(0, nSrcPts)
        matchedTgt, numpy.ndarray[[nMatchedPts,], int]: Indices for matched points in the range(0, nTgtPts)
    """
    # Check the nb of dimensions
    assert src.ndim == tgt.ndim and src.ndim == 2, "The pb must be at least 2D !"

    # Build the cost array which is the distance between a point in frame n and points in frame n+1
    D = spatial.distance_matrix(src, tgt, p=2) ** 2

    # Set distance above the threshold to infinity so that they can never generate a link
    # But we represent infinity as 2*dist_max+1 to allow a cost to be infinity.
    isAboveDmax = D > d_max**2
    fakeInf = 2 * D[isAboveDmax].max() + 1
    D[isAboveDmax] = fakeInf

    # Find the optimal assignment
    matchedSrc, matchedTgt = optimize.linear_sum_assignment(D)

    # Select matches according to fake infinite cost
    isNotInfCost = D[matchedSrc, matchedTgt] != fakeInf
    matchedSrc = matchedSrc[isNotInfCost]
    matchedTgt = matchedTgt[isNotInfCost]

    return matchedSrc, matchedTgt


def NearestNeighborLinker(
    src: numpy.ndarray[float, Literal["nSrcPts", "nDims"]],
    tgt: numpy.ndarray[float, Literal["nTgtPts", "nDims"]],
    d_max: float = numpy.inf,
) -> tuple[numpy.ndarray[Literal["nMatchedPts"], int], numpy.ndarray[Literal["nMatchedPts"], int]]:
    """
    NearestNeighborLinker paires a point of src and tgt variables while choosing the minimum distance at each iteration. If a distance exceeds d_max, then it is set to infinity so that the associated points cannot be matched.

    Args:
        * src, numpy.ndarray[float, [nSrcPts, nDims]]: Point matrix for a frame
        * tgt, numpy.ndarray[float, [nTgtPts, nDims]]: Point matrix for another frame
        * d_max, float: Distance threshold at which points are not paired

    Returns:
        matchedSrc, numpy.ndarray[[nMatchedPts,], int]: Indices for matched points in the range(0, nSrcPts)
        matchedTgt, numpy.ndarray[[nMatchedPts,], int]: Indices for matched points in the range(0, nTgtPts)
    """
    # Build the cost array which is the distance between a point in frame n and points in frame n+1
    distMatrix = spatial.distance_matrix(src, tgt)

    # Set distance above the threshold to infinity so that they can never generate a link
    distMatrix[distMatrix > d_max] = numpy.inf

    # Parse distance matrix
    matchedSrc = list()
    matchedTgt = list()
    while not numpy.isinf(distMatrix).all():
        tgtIndices = distMatrix.argmin(axis=1)

        for srcIdx in distMatrix[range(distMatrix.shape[0]), tgtIndices].argsort():
            tgtIdx = tgtIndices[srcIdx]

            # Did we already assigned this tgt to a src?
            if tgtIdx in matchedTgt:
                # Yes, then exit the loop and change the distance matrix to prevent this assignment
                break

            else:
                # No, then store this assignment
                matchedSrc.append(srcIdx)
                matchedTgt.append(tgtIdx)

                # And make it impossible to find it again by putting the tgt point to infinity in the distance matrix
                distMatrix[srcIdx, :] = numpy.inf
                distMatrix[:, tgtIdx] = numpy.inf

                # Exit loop if distMatrix is filled with inf
                if numpy.isinf(distMatrix).all():
                    break

    return matchedSrc, matchedTgt


def SimpleTracker(
    data: numpy.ndarray[Literal["nPts"],
                        numpy.dtype[tuple[tuple[Literal["pos"], float, Literal["nDims"]],
                                          tuple[Literal["frame_no"], int]]]],
    max_linking_dist: float = numpy.inf,
    max_gap_closing: float = 3,
    min_track_len: int = 2,
) -> numpy.ndarray[Literal["nPts"],
                   numpy.dtype[tuple[tuple[Literal["pos"], float, Literal["nDims"]],
                                     tuple[Literal["frame_no"], int],
                                     tuple[Literal["track_no"], int]]]]:
    """
    SimpleTracker is a simple implementation of a tracking algorithm, that can deal with gaps. A gap happens when one particle, that was detected in one frame, is not detected in the subsequent one. If not dealt with, this generates a track break, or a gap, in the frame where the particule disappears, and a false new track in the frame where it re-appears.

    It first does a frame-to-frame linking step, where links are first created between each frame pair, using a linear sum assignment optimization algorithm with the distance matrix as the cost matrix. Then a second iteration is done through the data, investigating track ends. If a track beginning is found close to a track end in a subsequent track, a link spanning multiple frame can be created, bridging the gap and restoring the track.
    Finally, the tracks are generated by searching for connected components of the matching graph and by filtering the tracks according to their lengths to return the list of points with a new field corresponding to the track number (-1=unmatched points).

    Args:
        * data, numpy.ndarray[numpy.dtype([('pos', float, (nDims,)), ('frame_no', int)]), [nPts,]]: List of points with their located position in the associated frame
        * max_linking_dist, float: Distance threshold at which points are not paired
        * max_gap_closing, int: Maximal frame distance in gap-closing => frames further way than this value will not be investigated for gap closing
        * min_track_len, int: Minimal track length which corresponds to the nb of points in the track

    Returns:
        matchedPts, numpy.ndarray[numpy.dtype([('pos', float, (nDims,)), ('frame_no', int), ('track_no', int)]), [nPts,]]: List of points with their located position in the associated frame and track
    """
    # Parse arguments
    max_linking_dist = float(max_linking_dist)
    max_gap_closing = float(max_gap_closing)
    min_track_len = int(min_track_len)
    points = numpy.vstack(data["pos"], dtype=float)
    frame_no = data["frame_no"].astype(int)
    frame_no_min = frame_no.min(axis=None)
    frame_no_max = frame_no.max(axis=None)
    assert frame_no_min >= 0, "The minimal frame number is negative !"
    assert min_track_len >= 2, "The minimal length for tracks should be positive !"
    iPts = numpy.arange(data.size)

    # Create a map for indexing frame index and their related points
    frameToPtsMap = sparse.coo_array(([1] * data.size, (frame_no, iPts)),
                                     shape=(frame_no_max + 1, data.size), dtype=bool).tolil()

    # Create sparse matching matrix which is a directed graph and 2 arrays for nodes with no child and no parent
    matchingGraph = sparse.lil_matrix((data.size, data.size), dtype=bool)

    # Frame pair matching
    for curFrame, nxtFrame in itertools.pairwise(range(frame_no_min, frame_no_max + 1)):
        # Get points for current and next frames
        iSrcPts = frameToPtsMap.rows[curFrame]
        iTgtPts = frameToPtsMap.rows[nxtFrame]
        src = points[iSrcPts, :]
        tgt = points[iTgtPts, :]

        # Skip this iteration if there is a frame with no points
        if src.size == 0 or tgt.size == 0:
            continue

        # Frame to frame linking
        matchedSrc, matchedTgt = HungarianLinker(src, tgt, max_linking_dist)
        matchedSrc = numpy.take(iSrcPts, matchedSrc)
        matchedTgt = numpy.take(iTgtPts, matchedTgt)
        matchingGraph[matchedSrc, matchedTgt] = 1

    # Gap closing
    hasNoChild = numpy.flatnonzero(matchingGraph.getnnz(axis=1) == 0)
    hasNoChild = sparse.coo_array(([True] * hasNoChild.size, (frame_no[hasNoChild], hasNoChild)), 
                                  shape=frameToPtsMap.shape, dtype=bool).tolil()
    hasNoParent = numpy.flatnonzero(matchingGraph.tocsc().getnnz(axis=0) == 0)
    hasNoParent = sparse.coo_array(([True] * hasNoParent.size, (frame_no[hasNoParent], hasNoParent)),
                                    shape=frameToPtsMap.shape, dtype=bool).tolil()
    for curFrame in range(frame_no_min, frame_no_max + 1):
        for nxtFrame in range(curFrame + 2, min(curFrame + int(max_gap_closing), frame_no_max) + 1):
            # Get unmatched points in source frame
            iSrcPts = hasNoChild.rows[curFrame]
            src = points[iSrcPts, :]

            # Get unmatched target points
            iTgtPts = hasNoParent.rows[nxtFrame]
            tgt = points[iTgtPts, :]

            # Skip this iteration if there is a frame with no points
            if src.size == 0 or tgt.size == 0:
                continue

            # Frame with gap matching
            matchedSrc, matchedTgt = NearestNeighborLinker(src, tgt, max_linking_dist)
            matchedSrc = numpy.take(iSrcPts, matchedSrc)
            matchedTgt = numpy.take(iTgtPts, matchedTgt)
            matchingGraph[matchedSrc, matchedTgt] = 1
            hasNoChild[curFrame, matchedSrc] = False
            hasNoParent[nxtFrame, matchedTgt] = False

    # Build tracks
    matchedPts = numpy.empty((data.size,), dtype=data.dtype.descr + [('track_no', int)])
    for fieldName in data.dtype.names:
        matchedPts[fieldName] = data[fieldName]
    matchedPts['track_no'] = -1
    _, labels = sparse.csgraph.connected_components(matchingGraph, directed=True, connection="weak", return_labels=True)
    trackIds, counts = numpy.unique(labels, return_counts=True)
    tracksNo = trackIds[counts >= min_track_len]
    isValidTrack = numpy.isin(labels, tracksNo)
    for iTrack, track_no in enumerate(tracksNo):
        labels[labels == track_no] = iTrack
    matchedPts["track_no"][isValidTrack] = labels[isValidTrack]

    return matchedPts
