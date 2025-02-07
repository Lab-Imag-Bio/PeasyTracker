# simpletracker - A simple particle tracking algorithm for MATLAB that can deal with gaps.


*Tracking* , or particle linking, consist in re-building the trajectories of one or several particles as they move along time. Their position is reported at each frame, but their identity is yet unknown: we do not know what particle in one frame corresponding to a particle in the previous frame. Tracking algorithms aim at providing a solution for this problem. 

`simpletracker.m` is - as the name says - a simple implementation of a tracking algorithm, that can deal with gaps. A gap happens when one particle that was detected in one frame is not detected in the subsequent one. If not dealt with, this generates a track break, or a gap, in the frame where the particle disappear, and a false new track in the frame where it re-appear. 

`simpletracker` first do a frame-to-frame linking step, where links are first created between each frame pair, using the hungarian algorithm of `hungarianlinker`. Links are created amongst particle paris found to be the closest (euclidean distance). By virtue of the hungarian algorithm, it is ensured that the sum of the pair distances is minimized over all particles between two frames. 

Then a second iteration is done through the data, investigating track ends. If a track beginning is found close to a track end in a subsequent track, a link spanning multiple frame can be created, bridging the gap and restoring the track. The gap-closing step uses the nearest neighbor algorithm provided by `nearestneighborlinker`. 

## INPUT SYNTAX 

`tracks = SIMPLETRACKER(points)` rebuilds the tracks generated by the particle whose coordinates are in |points|. |points| must be a cell array, with one cell per frame considered. Each cell then contains the coordinates of the particles found in that frame in the shape of a `n_points x n_dim` double array, where `n_points` is the number of points in that frame (that can vary a lot from one frame to another) and `n_dim` is the dimensionality of the problem (1 for 1D, 2 for 2D, 3 for 3D, etc...). 

`tracks = SIMPLETRACKER(points, max_linking_distance)` defines a maximal value in particle linking. Two particles will not be linked (even if they are the remaining closest pair) if their distance is larger than this value. By default, it is infinite, not preventing nay linking. 

`tracks = SIMPLETRACKER(points, max_linking_distance, max_gap_closing)` defines a maximal frame distance in gap-closing. Frames further way than this value will not be investigated for gap closing. By default, it has the value of 3. 

`track = SIMPLETRACKER(points, max_linking_distance, max_gap_closing, debug)` adds some printed information about the tracking process.

## OUTPUT SYNTAX 

`track = SIMPLETRACKER(...)` return a cell array, with one cell per found track. Each track is made of a `n_frames x 1` integer array, containing the index of the particle belonging to that track in the corresponding frame. `NaN` values report that for this track at this frame, a particle could not be found (gap). 

Example output: `track{1} = [ 1 2 1 NaN 4 ]` means that the first track is made of the particle 1 in the first frame, the particle 2 in the second frame, the particle 1 in the 3rd frame, no particle in the 4th frame, and the 4th particle in the 5th frame. 

`[ tracks adjacency_tracks ] = SIMPLETRACKER(...)` return also a cell array with one cell per track, but the indices in each track are the global indices of the concatenated points array, that can be obtained by `all_points = vertcat( points{:} );`. It is very useful for plotting applications. 

`[ tracks adjacency_tracks A ] = SIMPLETRACKER(...)` return the sparse adjacency matrix. This matrix is made everywhere of 0s, expect for links between a source particle (row) and a target particle (column) where there is a 1. Rows and columns indices are for points in the concatenated points array. Only forward links are reported (from a frame to a frame later), so this matrix has no non-zero elements in the bottom left diagonal half. Reconstructing a crude trajectory using this matrix can be as simple as calling `gplot( A, vertcat( points{:} ) )`

## Demo.

Check the `publish` version of the demo script in this repo:
http://htmlpreview.github.io/?https://raw.githubusercontent.com/tinevez/simpletracker/master/publish/html/TestSimpleTracker.html

## Cite as
Jean-Yves Tinevez (2024). simpletracker (https://github.com/tinevez/simpletracker), GitHub. Retrieved the 25th June 2024.
