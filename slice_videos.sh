#!/usr/bin/env bash
# use prc/videos/... instead for using pre-processed videos
for entry in "videos/fake"/*
do
    file = "$(basename -- $entry)"
    ffmpeg -i $entry -c copy -map 0 -segment_time 00:00:10 -f segment -reset_timestamps 1 "split/$entry%03d.mp4"
done
for entry in "videos/real"/*
do
    ffmpeg -i $entry -c copy -map 0 -segment_time 00:00:10 -f segment -reset_timestamps 1 "split/$entry%03d.mp4"
done