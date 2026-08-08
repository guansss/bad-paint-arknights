##### Goal

Build a script that simulates a series of touch events on an Android emulator to draw a video frame by frame in a painting application.

## Tech Stack

- **Python**: OpenCV, [scrcpy-client](https://github.com/leng-yue/py-scrcpy-client)
- **External Tools**: ffmpeg, ADB

## Steps

1. Connect to the Android emulator using ADB.
2. Capture the video frame by frame.
3. For each frame, process the image to determine the touch events needed to replicate the drawing.
4. Use scrcpy to send the touch events to the emulator.
5. Capture and save a screenshot of the emulator after each frame is drawn.
6. Compile the screenshots into a video to visualize the drawing process.

## Basis

#### Painting Application

The painting application is assumed to be already open on the emulator and ready for interaction.

The canvas is a square with 24x24 square cells, where each cell can be filled with a specific color. The canvas supports freehand drawing, i.e., a continuous touch event will create a continuous line on the canvas.

Colors can be selected from a palette.

There is a "clear" button that, with a confirmation dialog, clears the entire canvas.

## File Structure

```
project/
├── SPEC.md
├── assets/
│   ├── templates/
│   │   ├── palette-first-row.png
│   │   ├── clear-button.png
│   │   └── clear-confirm-button.png
│   └── video.mp4
├── debug/
├── screenshots/
├── main.py
├── requirements.txt
└── output.mp4
```

## Implementation Details

#### Environment

Use Python venv to manage dependencies.

The path of required external libraries (ffmpeg, ADB) can be specified through variables, and if not specified, the script will assume that they are available in the system's PATH.

#### Communication with the emulator

Use `scrcpy-client` to control scrcpy and connect to the Android emulator for sending touch events and reading screen through video stream.

The address of the emulator can be specified through a variable, and if not specified, let scrcpy automatically detect the emulator.

A short configurable per-action delay should be added after each action to ensure that the emulator and the app have enough time to render before the next action.

#### Detecting interaction areas

Before drawing, detect the interaction areas through the video stream.

To detect the canvas, find the largest white square in the screen. To speed up the process, assume the canvas's side length to be at least 50% of the screen height. The canvas is "almost" uniformly white, but not perfectly white as it contains a subtle light gray background pattern.

To detect the color palette, use template matching with `palette-first-row.png` to find the location of the palette. The `palette-first-row.png` image contains the first row of the palette, which is 4 grids of colors, where the leftmost grid is black and the rightmost grid is white, which are the only colors that will be used for drawing.

To detect the "clear" button, use template matching with `clear-button.png` to find the location of the button.

Each time after pressing the "clear" button, use template matching with `clear-confirm-button.png` to detect the "Confirm" button.

If any of the template matching fails, the script should raise an error and stop execution.

The emulator's screen may have a different resolution than the one that the template images were created with (1920x1080), so the template matching should be done with multi-scale template matching to account for different resolutions.

#### Drawing

Before each run, clear the canvas by pressing the "clear" button and confirming the action, and clear the `screenshots/` directory to remove any previous screenshots.

For each frame, scale the image to fit the canvas' resolution simulating the behavior of `object-fit: cover` CSS property. This means that the image will be scaled to cover the entire canvas while maintaining its aspect ratio, potentially cropping parts of the image that exceed the canvas dimensions.

The image will be thresholded into a binary image to determine which cells should be filled with black and which should remain white.

The image will be drawn on the canvas in a scan line order, i.e., from the top-left cell to the bottom-right cell, row by row.

For an isolated pixel, simulate a tap event at the center of the corresponding cell. For a continuous line of pixels in the same row, simulate a touch event that starts at the first cell and slides to the last cell in that line.

To optimize the drawing process, before drawing each frame, compare the current frame with the previous frame to determine which cells have changed. If the changed pixels are less than the pixels that are supposed to be drawn for the current frame, only draw the changed cells by swapping their colors between black and white. Otherwise, clear the canvas first (by pressing the "clear" button and confirming) and then draw all the pixels for the current frame.

To pick a color, simulate a tap event on the corresponding color grid in the palette. For black, tap on the leftmost grid of the palette, and for white, tap on the rightmost grid. Since the canvas's background is already white, only the black pixels need to be drawn, unless the previous frame had black pixels that need to be overwritten with white.

After drawing each frame, capture a frame of the video stream and save it as a screenshot in the `screenshots/` directory.

#### Video Processing

The video is read from the first video file in the `assets/` directory. The targeting format is MP4, and other formats are supported as long as the format is fully compatible with our MP4 workflow and does not require additional code to handle.

The source video should be resampled to a configured frame rate (default: 5 fps).

After all frames have been drawn, compile the screenshots into a video and save it as `output.mp4`.

The output video should be encoded in H.264 format, with the source video's audio (or silence if the source video has no audio).
