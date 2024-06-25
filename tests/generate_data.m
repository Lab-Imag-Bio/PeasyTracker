%% Generate data
n_dim = 2;
n_points_per_frame = 10; 
n_frames = 200;
points = cell(n_frames, 1);

% Random start position
start = 20 * rand(n_points_per_frame, n_dim);

% Span initial direction
theta = linspace(0, 2* pi/4, n_points_per_frame)';
vec = [ cos(theta) sin(theta) ];

% Random direction change
theta_increase = pi / n_frames * rand(n_points_per_frame, 1);

for i_frame = 1 : n_frames 
   
    % Disperse points as if their position was increasing by 1.5 in average
    % each frame.
    frame_points = start + vec .* i_frame .* [cos(theta_increase * i_frame) sin(theta_increase * i_frame) ] + rand(n_points_per_frame, n_dim) ;
    
    % Randomize them;
    randomizer = rand(n_points_per_frame, 1);
    [ sorted, index ] = sort(randomizer);
    frame_points = frame_points(index, :);
    
    % Delete some of them, possible
    deleter = randn;
    while (deleter > 0);
        frame_points(1, :) = [];
        deleter = deleter - 1;
    end
    
    points{i_frame} = frame_points;
    
end

%% Apply SimpleTracker
tic
max_linking_distance = 4;
max_gap_closing = 10;
min_track_len = 3;
debug = true;
[ tracks, adjacency_tracks, A ] = simpletracker(points,...
    'MaxLinkingDistance', max_linking_distance, ...
    'MaxGapClosing', max_gap_closing, ...
    'Debug', false);
toc

%% Export data
tracks_py = repmat(struct('pos', [], 'frame_no', [], 'track_no', []), size(A, 1), 1);
iPt = 1;
for iFrame=1:numel(points)
  for iPoint=1:size(points{iFrame}, 1)
    tracks_py(iPt) = struct('pos', points{iFrame}(iPoint, :), 'frame_no', iFrame, 'track_no', -1);
    iPt = iPt + 1;
  end
end

for iTrack=1:numel(adjacency_tracks)
  for iPt=1:numel(adjacency_tracks{iTrack})
    tracks_py(adjacency_tracks{iTrack}(iPt)).track_no = iTrack-1;
  end
end

save trackerData.mat -v6 tracks_py A max_gap_closing max_linking_distance
