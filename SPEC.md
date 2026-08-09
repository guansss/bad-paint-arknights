##### Goal

Build a script that simulates a series of touch events on an Android emulator to draw a video frame by frame in a painting application.

## Tech Stack

- **Python**: OpenCV and the Python standard library
- **External Tools**: ffmpeg and ADB
- **Device Touch Service**: [minitouch](https://github.com/openstf/minitouch), with ABI-specific binaries kept in the repository

## Steps

1. Connect to the Android emulator using ADB.
2. Capture the video frame by frame.
3. For each frame, process the image to determine the touch events needed to replicate the drawing.
4. Use minitouch to send the touch events to the emulator.
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
│   ├── minitouch/
│   │   └── <abi>/
│   │       └── minitouch
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

Use a Python virtual environment to manage dependencies.

The paths of ADB and ffmpeg can be configured. When omitted, the script assumes they are available on the system PATH.

Minitouch binaries must be present under `assets/minitouch/<abi>/minitouch`, where `<abi>` exactly matches the device's `ro.product.cpu.abi` value. For example, an `x86_64` emulator requires `assets/minitouch/x86_64/minitouch`.

#### Communication with the emulator

ADB is used to connect to the emulator and capture screenshots. Every screen read runs `adb exec-out screencap -p`; OpenCV decodes the returned PNG into a BGR frame. Each successful capture receives a monotonically increasing sequence number, so validation reads always use a newly captured screenshot rather than a buffered video frame.

The emulator serial can be configured. If omitted and multiple devices are online, the script selects the first device reported by `adb devices` and logs the selection.

At startup, the script reads the emulator ABI, pushes the matching minitouch binary to `/data/local/tmp/minitouch`, makes it executable, starts it through ADB, and forwards its abstract local socket to the configured host port. The minitouch capability handshake supplies the device touch-coordinate range; screen coordinates from screenshots are normalized to that range before each touch command.

A short configurable per-action delay should be added after each action to ensure that the emulator and the app have enough time to render before the next action.

The transport is fail-fast. ADB screenshot, PNG decode, minitouch provisioning, socket-forwarding, handshake, or gesture-command failures stop the run with an actionable error. There is no fallback touch or capture backend. Shutdown closes the minitouch socket, removes the ADB forward, and stops the minitouch process.

#### Configuration

The active touch and capture configuration includes:

- `adb_path` and `emulator_serial` for selecting the Android device.
- `minitouch_binaries_dir` for the root containing ABI-specific minitouch binaries.
- `minitouch_host_port` for the local ADB-forwarded minitouch port.
- `minitouch_tap_hold_sec` and `minitouch_swipe_step_delay_sec` for reliable gesture timing.
- `minitouch_swipe_step_px` for the screenshot-pixel distance between interpolated swipe moves; the default is `24`.
- `frame_wait_timeout_sec` as the maximum duration of an individual ADB screencap command.

#### Detecting interaction areas

Before drawing, detect the interaction areas from an ADB screenshot.

To detect the canvas, find the largest white square in the screen. To speed up the process, assume the canvas's side length to be at least 50% of the screen height. The canvas is "almost" uniformly white, but not perfectly white as it contains a subtle light gray background pattern.

To detect the color palette, use template matching with `palette-first-row.png` to find the location of the palette. The `palette-first-row.png` image contains the first row of the palette. The first row of the palette contains 4 grids of colors - black, light gray, beige, and white.

To detect the "clear" button, use template matching with `clear-button.png` to find the location of the button.

Each time after pressing the "clear" button, use template matching with `clear-confirm-button.png` to detect the "Confirm" button.

If any of the template matching fails, the script should raise an error and stop execution.

The emulator's screen may have a different resolution than the one that the template images were created with (1920x1080), so the template matching should be done with multi-scale template matching to account for different resolutions.

#### Drawing

Before each run, clear the canvas by pressing the "clear" button and confirming the action, and clear the `screenshots/` directory to remove any previous screenshots.

For each frame, scale the image to fit the canvas' resolution simulating the behavior of `object-fit: cover` CSS property. This means that the image will be scaled to cover the entire canvas while maintaining its aspect ratio, potentially cropping parts of the image that exceed the canvas dimensions.

The image will be converted to gray scale before drawing. Only three colors will be used for drawing: black, light gray, and white (i.e., the 1st, 2nd, and 4th colors in the palette). To determine the color of each pixel, first extract the candidate colors' actual RGB values from the detected palette, then find the closest color to the pixel's color in the frame.

The image will be drawn on the canvas in a scan line order, i.e., from the top-left cell to the bottom-right cell, row by row.

For an isolated pixel, simulate a tap event at the center of the corresponding cell. For a continuous line of pixels in the same row, simulate a touch event that starts at the first cell and slides to the last cell in that line.

To optimize the drawing process, before drawing each frame, compare the current frame with the previous frame to determine which cells have changed. If the changed pixels are less than the pixels that are supposed to be drawn for the current frame, only draw the changed cells. Otherwise, clear the canvas first (by pressing the "clear" button and confirming) and then draw all the pixels for the current frame.

To pick a color, simulate a tap event on the corresponding color grid in the palette. Since the canvas's background is already white, only the non-white pixels need to be drawn, unless the previous frame had non-white pixels that need to be overwritten with white.

After drawing each frame, capture a fresh ADB screenshot and save it in the `screenshots/` directory.

#### Video Processing

The video is read from the first video file in the `assets/` directory. The targeting format is MP4, and other formats are supported as long as the format is fully compatible with our MP4 workflow and does not require additional code to handle.

The source video should be resampled to a configured frame rate (default: 5 fps).

After all frames have been drawn, compile the screenshots into a video and save it as `output.mp4`.

The output video should be encoded in H.264 format, with the source video's audio (or silence if the source video has no audio).
