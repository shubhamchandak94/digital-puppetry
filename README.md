# Pose Animator for STEM to SHTEM

A twist on the conventional video streaming pipeline for the purposes of reductions in net latency and bandwidth consumption. 

**arXiv preprint:** https://arxiv.org/abs/2011.03800

## Install, Build, Run

Do a recursive download:
```sh
git clone -b  --recursive https://github.com/shubhamchandak94/digital-puppetry
```

Install dependencies and prepare the build directory:

```sh
yarn
```

To set up the connection between two machines, first check that the signaling server is active by
opening https://vast-earth-73765.herokuapp.com/ on your browser. Note that the code for the signaling
server is available at https://github.com/shubhamchandak94/webrtc-signaling-server and the URL is 
hardcoded in camera.js. Note that this is currently implemented only for the animation stream. If 
you want only one-way communication, select the doNotTransmit checkbox on one machine.

Launch a live dev server while watching for changes:

```sh
yarn watch
```
Do this on two machines (or two tabs on same machine) and press Connect to start the stream.

The following might be useful if you face issues with the steps above:
- https://github.com/nodejs/node-gyp/issues/1927#issuecomment-661825834
- https://github.com/yemount/pose-animator/issues/12#issuecomment-626741765

The second link above no longer allows you to see the original script in chrome which is quite useful for debugging.

A better solution would be to NOT add --no-source-maps to package.json and instead replace the file `node_modules/paper/dist/paper-full.js` with a working copy available in this repository [paper-full.js](paper-full.js) by running (after `yarn` command):
```
cp paper-full.js node_modules/paper/dist/paper-full.js
```

## Platform support

Demos are supported on Desktop Chrome and iOS Safari.

It should also run on Chrome on Android and potentially more Android mobile browsers though support has not been tested yet.


## README from original Pose Animator work

Pose Animator takes a 2D vector illustration and animates its containing curves in real-time based on the recognition result from PoseNet and FaceMesh. It borrows the idea of skeleton-based animation from computer graphics and applies it to vector characters.

This is running in the browser in realtime using [TensorFlow.js](https://www.tensorflow.org/js). Check out more cool TF.js demos [here](https://www.tensorflow.org/js/demos).

*This is not an officially supported Google product.*

<img src="/resources/gifs/avatar-new-1.gif?raw=true" alt="cameraDemo" style="width: 250px;"/>
<img src="/resources/gifs/avatar-new-full-body.gif?raw=true" alt="cameraDemo" style="width: 250px;"/>

In skeletal animation a character is represented in two parts:
1. a surface used to draw the character, and 
1. a hierarchical set of interconnected bones used to animate the surface. 

In Pose Animator, the surface is defined by the 2D vector paths in the input SVG files. For the bone structure, Pose Animator provides a predefined rig (bone hierarchy) representation, designed based on the keypoints from PoseNet and FaceMesh. This bone structure’s initial pose is specified in the input SVG file, along with the character illustration, while the real time bone positions are updated by the recognition result from ML models.

<img src="https://firebasestorage.googleapis.com/v0/b/pose-animator-demo.appspot.com/o/ml-keypoints.png?alt=media" style="width:250px;"/>

<img src="/resources/gifs/avatar-new-bezier-1.gif?raw=true" alt="cameraDemo" style="width: 250px;"/>

// TODO: Add blog post link.
For more details on its technical design please check out this blog post.

### Demo 1: [Camera feed](https://pose-animator-demo.firebaseapp.com/camera.html)

The camera demo animates a 2D avatar in real-time from a webcam video stream.


### Demo 2: [Static image](https://pose-animator-demo.firebaseapp.com/static_image.html)

The static image demo shows the avatar positioned from a single image.



# Animate your own design

1. Download the [sample skeleton SVG here](/resources/samples/skeleton.svg).
1. Create a new file in your vector graphics editor of choice. Copy the group named ‘skeleton’ from the above file into your working file. Note: 
	* Do not add, remove or rename the joints (circles) in this group. Pose Animator relies on these named paths to read the skeleton’s initial position. Missing joints will cause errors.
	* However you can move the joints around to embed them into your illustration. See step 4.
1. Create a new group and name it ‘illustration’, next to the ‘skeleton’ group. This is the group where you can put all the paths for your illustration.
    * Flatten all subgroups so that ‘illustration’ only contains path elements.
    * Composite paths are not supported at the moment.
    * The working file structure should look like this:
	```
        [Layer 1]
        |---- skeleton
        |---- illustration
              |---- path 1
              |---- path 2
              |---- path 3
	```
1. Embed the sample skeleton in ‘skeleton’ group into your illustration by moving the joints around.
1. Export the file as an SVG file.
1. Open [Pose Animator camera demo](https://pose-animator-demo.firebaseapp.com/camera.html). Once everything loads, drop your SVG file into the browser tab. You should be able to see it come to life :D
