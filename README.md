# PeasyTracker - An adaptation of SimpleTracker in Python

![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)

*Tracking*, or particle linking, consists in building the particles' trajectory as they move along time. The particles are localized at each frames but not identified ! This tracking algorithm aims at solving this problem.

`PeasyTracker` is a Python implementation of the Matlab addon `SimpleTracker` developed by Jean-Yves Tinevez and is also able to deal with gaps. A gap happens when a particle that was detected in one frame is not detected in the next one. If not dealt with, this generates 2 short tracks. 

`PeasyTracker` first tries to link particles across 2 consecutive frames (*frame_1* and *frame_2*) using `HungarianLinker` function. This function computes the euclidean distance matrix between particles in *frame_1* and the others in *frame_2*; and uses it as the cost matrix for the linear sum assignment optimization algorithm which minimizes the overall sum of the euclidean distances paired points.

Then it tries from the created matching matrix to link a track end to another track beginning that are close temporally and spatially using `NearestNeighborLinker` function. This allows to bridge the gaps in the matching matrix and to restore the track.

Finally, the tracks identification are generated by searching for connected components of the matching graph and by filtering the tracks according to their lengths. The unmatched points are designated by `-1` in the `track_no` field.

## Installation
The [Github repository](https://github.com/Lab-Imag-Bio/PeasyTracker) was published on [PyPI · The Python Package Index](https://pypi.org/) site. In this way, ``PeasyTracker` can be simply installed using the following command:
```bash
pip install PeasyTracker
```
## API

`SimpleTracker` is the main function of `PeasyTracker` library.

```python
from peasyTracker import SimpleTracker

tracks = SimpleTracker(data=localizedPts, 
                       max_linking_dist=maxDistBtwnPairedPts,
                       max_gap_closing = maxGap,
                       min_track_len = minTrackLen)
```

### Inputs
* `data` is a numpy structured array with:
    * the field `pos` designating the particle position as a 1D array,
    * the field `frame_no` designating the frame number where the particle was localized.

* `max_linking_dist` is the maximal distance from which 2 particles will not be linked if they're separated by a greater distance. Its default value is infinite, thus not preventing any linking. 

* `max_gap_closing` is the maximal size of a gap in terms of frames so that the linking of a beginning track and another track end will not be investigated if the difference between their associated frame number is greater than this value. By default, it has the value of 3. 

* `min_track_len` is the minimal track length so that the tracks, for which the number of matched points is smaller than this value, will be removed and their associated points will be considered as unmatched. By default, the minimal track length is 2.

### Output
* `track` is a numpy structured array which is a copy of `data` input but with the extra field `track_no` corresponding to a track id. The value of this field for an associated point is -1 as the point was unmatched.

## Corresponding authors
Codes: [Jacques Battaglia](mailto:jacques.battaglia@cnrs.fr), [Jean-Baptiste Deloges](mailto:jean-baptiste.deloges@sorbonne-universite.fr)

Materials, collaborations, rights and others: [Olivier Couture](mailto:olivier.couture@sorbonne-universite.fr)

Code Available under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)  

Laboratoire d'Imagerie Biomédicale, Team PPM. 15 rue de l'Ecole de Médecine, 75006, Paris, France.
CNRS, Sorbonne Université, INSERM

## Disclaimer
THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS AND CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

by JB, J-BD, OC.