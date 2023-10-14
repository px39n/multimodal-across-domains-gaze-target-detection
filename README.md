# Multimodal Across Domains Gaze Target Detection
Official PyTorch implementation of ["Multimodal Across Domains Gaze Target Detection"](https://dl.acm.org/doi/10.1145/3536221.3556624) at [ICMI 2022](https://icmi.acm.org/2022/).
![An image of our neural network](/assets/network.png?raw=true)

## Requirements
### Environment
To run this repo create a new conda environment and configure all environmental variables using the provided templates.

```bash
conda env create -f environment.yml

cp .env.example .env
nano .env
```

Due to the complexity of the network use a recent NVidia GPU with at least 6GB of memory available and CUDA 11.3+ installed.
Also, we suggest running everything on a Linux-based OS, preferably Ubuntu 20.04.

### Demo
 
1. Video properties such as frame count and FPS are extracted using OpenCV.
2. Based on the original video's FPS and the desired `sampling_fps`, the function calculates the frequency at which frames should be extracted.
3. The frames are then saved to a directory which has the same name as the video file (minus its extension) and is located in the same parent directory as the video.
4. Each saved frame is processed using either the `predict_image_with_annotation` or `predict_image_without_annotation` function, depending on the value of the `annotated` flag.
5. After processing all frames, the face has been detected autmatically, and annotation has been saved in style of gazefollow. The depth image has been process automatically.
6. Function compiles them back into a video file and saves it to the output directory.

**Output**:
The function will save the processed video in the same parent directory as the original video. The output video's name will be appended with "_output.avi".

**Example Usage**:
```python
predict_video(r"D:\Datasets\engagement_follow\Video\00000.avi", startbox=[267, 200, 314, 248], sampling_fps=2, annotated=True)
```
In this example, the function will process the video located at `"D:\Datasets\engagement_follow\Video\00000.avi"`, extracting frames at an effective rate of 2 FPS, and using the given startbox for predictions. Since `annotated` is set to `True`, the function will use `predict_image_with_annotation` for processing frames.

![](https://i.imgur.com/3pdnmLF.png)

## API
