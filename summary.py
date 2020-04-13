import os
import argparse
import cv2
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np


def extract_patches(img, patch_size=64):
    return extract_patches_2d(
        image=img,
        patch_size=(patch_size, patch_size),
        max_patches=100,
        random_state=3,
    )


def load_img(img_path):
    return cv2.imread(img_path)


def load_frames(video_path, frame_step=24):
    frames = []
    count = 0

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frames.append(image)

    while success:
        count += 1
        success, image = vidcap.read()
        if success and count % frame_step == 0:
            frames.append(image)

    return frames, int(vidcap.get(cv2.CAP_PROP_FPS))


def extract_histogram(img):
    return cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])


def hist_intersect(hist_a, hist_b):
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_INTERSECT)


def compose_summary_video(output_filename, frames, change_indices, fps=24, window_size=64):
    frame_ranges = [list(range(
        idx - int(window_size / 2),
        idx + int(window_size / 2)
    )) for idx in change_indices]

    frames_to_write = [frame_index for sublist in frame_ranges for frame_index in sublist]

    os.system('mkdir tmp; rm -f ./tmp/*')
    frames_written = 0
    for i, frame in enumerate(frames):
        if i in frames_to_write:
            print(f'writing tmp/img{str(frames_written).zfill(5)}.png')
            cv2.imwrite(f'tmp/img{str(frames_written).zfill(5)}.png', frame)
            frames_written += 1
    os.system(f'ffmpeg -r {fps} -i tmp/img%05d.png {output_filename}')


def detect_changes(histograms, change_threshold=3000.0):
    avg_intersections = []
    first_frame = second_frame = None
    intersections = []
    for i, hist in enumerate(histograms):
        if not first_frame:
            first_frame = hist
        else:
            second_frame = hist
            avg, intersection_values = compare_frames(first_frame, second_frame)
            avg_intersections.append({
                'frame_index': i,
                'intersect': avg
            })
            # first_frame = second_frame = None
            first_frame = second_frame
            intersections += intersection_values
    
    under_threshold = [i['frame_index'] for i in avg_intersections if i['intersect'] < change_threshold]

    return under_threshold


def compare_frames(frame_a, frame_b):
    # compare histogram intersects of all patches in the given frames
    # returns average intersect
    patch_intersections = [hist_intersect(patch_hist, frame_b[i]) for i, patch_hist in enumerate(frame_a)]
    return sum(patch_intersections) / len(patch_intersections), patch_intersections


def main():
    args = get_args()
    input_video = args.input_video
    change_threshold = args.change_threshold

    frames, fps = load_frames(input_video)

    patch_size=64
    patches = [extract_patches(frame, patch_size=patch_size) for frame in frames]
    histograms = [[extract_histogram(patch) for patch in frame_patches] for frame_patches in patches]

    change_indices = detect_changes(histograms, (1.0 - change_threshold) * (patch_size * patch_size))

    output_video = f'summary_thresh_{change_threshold}_{input_video}'
    compose_summary_video(output_video, frames, change_indices, fps=fps)


def get_args():
    parser = argparse.ArgumentParser(description='Program to summarize a video by finding sequences with the highest changes.')
    parser.add_argument('input_video', help='Input video file path')
    parser.add_argument('change_threshold', type=float, help='Threshold to determine how big a change is considered to be a change, between 0.0 - 1.0 (small - big change)')
    return parser.parse_args()


if __name__ == "__main__":
    main()
