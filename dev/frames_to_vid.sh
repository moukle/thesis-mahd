#! /bin/sh

ffmpeg  -r 2 -pattern_type glob -i "*.jpg" -crf 25 -pix_fmt yuv420p video.mp4
