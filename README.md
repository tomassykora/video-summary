# Automatic Video Summarization
This program creates a short summarized video of a long input by taking only the frames from the original video in which the biggest changes are measured.

The inter-frame changes are detected by splitting each video frame into several image patches each of which is used to compute a histogram. Histogram intersections are computed between individual frames. Average of individual patch histogram intersections compared to the given input threshold.

## Usage

`python3 summary.py police-room-interview.mp4 3000.0 summarized-video.mp4`
