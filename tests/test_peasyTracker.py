from scipy.io import loadmat
from src.peasyTracker import SimpleTracker

def test_SimpleTracker():
    dataTest = loadmat("./tests/trackerData.mat")
    tracks = dataTest["tracks_py"].squeeze()
    max_gap_closing = dataTest['max_gap_closing']
    max_linking_dist = dataTest["max_linking_distance"]
    points = tracks[['pos', 'frame_no']]

    # Track them
    matchedPts = SimpleTracker(data=points, max_linking_dist=max_linking_dist, max_gap_closing=max_gap_closing.item())
    assert (matchedPts['track_no'] == tracks['track_no']).all()
