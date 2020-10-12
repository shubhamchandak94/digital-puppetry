/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as posenet_module from '@tensorflow-models/posenet';
import * as facemesh_module from '@tensorflow-models/facemesh';
import * as tf from '@tensorflow/tfjs';
import * as paper from 'paper';
import dat from 'dat.gui';
import Stats from 'stats.js';
import 'babel-polyfill';

import {
    drawKeypoints,
    drawPoint,
    drawSkeleton,
    isMobile,
    toggleLoadingUI,
    setStatusText,
} from './utils/demoUtils';

import {SVGUtils} from './utils/svgUtils';
import {PoseIllustration} from './illustrationGen/illustration';
import {Skeleton, facePartName2Index} from './illustrationGen/skeleton';
import {FileUtils} from './utils/fileUtils';


import * as girlSVG from './resources/illustration/girl.svg';
import * as boySVG from './resources/illustration/boy.svg';
import * as abstractSVG from './resources/illustration/abstract.svg';
import * as blathersSVG from './resources/illustration/blathers.svg';
import * as tomNookSVG from './resources/illustration/tom-nook.svg';

// signaling server
const HOST = 'wss://vast-earth-73765.herokuapp.com/';

// Camera stream video element
let video;
let videoWidth = 500;
let videoHeight = 500;

// Canvas
let faceDetection = null;
let illustration = null;
let canvasScope;
let canvasWidth = 500;
let canvasHeight = 500;

// ML models
let facemesh;
let posenet;
let minPoseConfidence = 0.15;
let minPartConfidence = 0.1;
let nmsRadius = 30.0;

// UI variables
var connectButton = null;

// Signalling Variables (used to communicate via server)
var uuid;
var serverConnection;

// RTC Variables!!
var peerConnection = null;  // RTCPeerConnection
var dataChannel = null;     // RTCDataChannel
var totalLatency = document.getElementById('total-latency');
var transmissionLatency = document.getElementById('transmission-latency');
var extractionLatency = document.getElementById('extraction-latency');
var renderLatency = document.getElementById('projection-latency');

// variable for received webrtc message
var WebRTCmessage;

// Misc
let mobile = false;
const stats = new Stats();
const avatarSvgs = {
    'girl': girlSVG.default,
    'boy': boySVG.default,
    'abstract': abstractSVG.default,
    'blathers': blathersSVG.default,
    'tom-nook': tomNookSVG.default,
};

// references for render setup
const keypointCanvas = document.getElementById('keypoints');
const canvas = document.getElementById('output');
const keypointCtx = keypointCanvas.getContext('2d');
const videoCtx = canvas.getContext('2d');

// WebRTC streaming channel
let channel;

// Analysis monitors
// const monitors = ['bytesReceived', 'packetsReceived', 'headerBytesReceived', 'packetsLost', 'totalDecodeTime', 'totalInterFrameDelay', 'codecId'];
const monitors = ['bytesReceived'];

// order list for poses deconstruction and reconstruction
const parts = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'];

// summations for finding necessary statistics
let previousTime;
let previousBytesIntegral = 0;

// in: buffer for a Uint32Array
function reconstructFaceData(positionsBuffer) {
    let view = new Float32Array(positionsBuffer);
    let out = [];

    view.forEach(coordinate => {
        out.push(coordinate);
    });
    return out;
}

/**
 * Loops the transmission of deconstructed poses
 *
 */
async function transmit() {
    var state = dataChannel.readyState;
    if (state !== 'open') {
        return;
    }
    if (guiState.debug.doNotTransmit === true) {
        document.getElementById('warningDoNotTransmit').innerText =
            'DoNotTransmit is ON, refresh to turn OFF.';
        return;
    }

    // Begin monitoring code for frames per second
    stats.begin();

    // measures latency starting from the before pose,mesh extraction to after rendering
    let beforeStamp = new Date().getTime();

    // get face information
    const input = tf.browser.fromPixels(canvas);
    faceDetection = await facemesh.estimateFaces(input, false, false);
    input.dispose();

    // initializes poses
    let poses = [];

    // populates poses
    let all_poses = await posenet.estimatePoses(video, {
        flipHorizontal: true,
        decodingMethod: 'multi-person',
        maxDetections: 1,
        scoreThreshold: minPartConfidence,
        nmsRadius: nmsRadius,
    });

    // merges all poses
    poses = poses.concat(all_poses);

    // clears previous render
    videoCtx.clearRect(0, 0, videoWidth, videoHeight);

    // draw video
    videoCtx.save();
    videoCtx.scale(-1, 1);
    videoCtx.translate(-videoWidth, 0);
    videoCtx.drawImage(video, 0, 0, videoWidth, videoHeight);
    videoCtx.restore();

    // projects pose and face onto svg
    keypointCtx.clearRect(0, 0, videoWidth, videoHeight);
    if (guiState.debug.showDetectionDebug) {
        poses.forEach(({score, keypoints}) => {
            if (score >= minPoseConfidence) {
                drawKeypoints(keypoints, minPartConfidence, keypointCtx);
                drawSkeleton(keypoints, minPartConfidence, keypointCtx);
            }
        });
        faceDetection.forEach(face => {
            for (let i = 0; i < face.scaledMesh.length; i++) {
                let p = face.scaledMesh[i];
                drawPoint(keypointCtx, p[1], p[0], 2, 'red');
            }
        });
    }

    // converts pose to streamable buffers
    let deconstructedPose;
    if (poses.length >= 1)
        deconstructedPose = deconstructPose(poses[0]);
    else
        deconstructedPose = null;

    // measure data collection latency
    let afterExtractionStamp = new Date().getTime();
    let extractionTime = afterExtractionStamp - beforeStamp;
    extractionLatency.innerText = `Extraction latency: ${extractionTime}ms`;

    // deconstructedPose === null if difference between consecutive frames is 0
    if (deconstructedPose !== null) {
        dataChannel.send(deconstructedPose[0].buffer);
        dataChannel.send(deconstructedPose[1].buffer);
    } else {
        dataChannel.send(0);
        dataChannel.send(0);
    }

    if (faceDetection && faceDetection.length > 0) {
        let face = Skeleton.toBufferedFaceFrame(faceDetection[0]);
        dataChannel.send(face.positions.buffer);
        dataChannel.send(face.faceInViewConfidence);
    } else {
        dataChannel.send(0);
        dataChannel.send(0);
    }

    //send before timestamp for pipeline latency measurements (total, transmission)
    dataChannel.send(beforeStamp);
    dataChannel.send(afterExtractionStamp);

    // End monitoring code for frames per second
    stats.end();

    // loop back
    setTimeout(transmit, 10);
}

// Set things up, connect event listeners, etc.

function startup() {
    // Get the local UI elements ready
    connectButton = document.getElementById('connectButton');

    // Set event listeners for user interface widgets

    connectButton.addEventListener('click', connect, false);

    // And set up connection to our websocket signalling server

    uuid = createUUID();

    serverConnection = new WebSocket(HOST);
    serverConnection.onmessage = gotMessageFromServer;
    serverConnection.onclose = function(event) {
        console.log(event);
        console.log('WebSocket is closed now.');
        document.getElementById('warningWebSocket').innerText =
            'WebSocket connection failed, possibly because of timeout or >2 clients, refresh to try again.';
    };
}

// Called when we initiate the connection

function connect() {
    console.log('connect');
    start(true);
}

// Start the WebRTC Connection
// We're either the caller (when we click 'connect' on our page)
// Or the receiver (when the other page clicks 'connect' and we recieve a signalling message through the websocket server)

function start(isCaller) {
    peerConnection = new RTCPeerConnection({});
    peerConnection.onicecandidate = gotIceCandidate;

    // If we're the caller, we create the Data Channel
    // Otherwise, it opens for us and we receive an event as soon as the peerConnection opens
    if (isCaller) {
        dataChannel = peerConnection.createDataChannel('pose-animator data channel');
        dataChannel.onopen = handleDataChannelStatusChange;
        dataChannel.onclose = handleDataChannelStatusChange;
    } else {
        peerConnection.ondatachannel = handleDataChannelCreated;
    }

    // Kick it off (if we're the caller)
    if (isCaller) {
        peerConnection.createOffer()
            .then(offer => peerConnection.setLocalDescription(offer))
            .then(() => console.log('set local offer description'))
            .then(() => serverConnection.send(JSON.stringify({
                'sdp': peerConnection.localDescription,
                'uuid': uuid,
            })))
            .then(() => console.log('sent offer description to remote'))
            .catch(errorHandler);
    }
}

// Handle messages from the Websocket signalling server
function gotMessageFromServer(message) {
    // If we haven't started WebRTC, now's the time to do it
    // We must be the receiver then (ie not the caller)
    if (!peerConnection) start(false);

    var signal = JSON.parse(message.data);

    // Ignore messages from ourself
    if (signal.uuid === uuid) return;

    console.log('signal: ' + message.data);

    if (signal.sdp) {
        peerConnection.setRemoteDescription(new RTCSessionDescription(signal.sdp))
            .then(() => console.log('set remote description'))
            .then(function() {
                // Only create answers in response to offers
                if (signal.sdp.type === 'offer') {
                    console.log('got offer');

                    peerConnection.createAnswer()
                        .then(answer => peerConnection.setLocalDescription(answer))
                        .then(() => console.log('set local answer description'))
                        .then(() => serverConnection.send(JSON.stringify({
                            'sdp': peerConnection.localDescription,
                            'uuid': uuid,
                        })))
                        .then(() => console.log('sent answer description to remote'))
                        .catch(errorHandler);
                }
            })
            .catch(errorHandler);
    } else if (signal.ice) {
        console.log('received ice candidate from remote');
        peerConnection.addIceCandidate(new RTCIceCandidate(signal.ice))
            .then(() => console.log('added ice candidate'))
            .catch(errorHandler);
    }
}

function gotIceCandidate(event) {
    if (event.candidate != null) {
        console.log('got ice candidate');
        serverConnection.send(JSON.stringify({
            'ice': event.candidate,
            'uuid': uuid,
        }));
        console.log('sent ice candiate to remote');
    }
}

function handleDataChannelReceiveMessage(event) {

    // for messages received, parse the transmitted arrays as poses and facemeshes and project them
    WebRTCmessage.push(event.data);

    // message s
    //
    //
    //
    // tructure:
    // [0, 1]: pose, [2]: mesh points, [3]: mesh confidence, [4]: pipelineInit timestamp,
    // [5]: timestamp after extraction before transmission
    if (WebRTCmessage.length === 6) {

        // record transmission time
        let afterExtractionStamp = new Date().getTime();
        transmissionLatency.innerText = `Transmission latency: ${afterExtractionStamp - WebRTCmessage[5]}ms`;
        
        if (WebRTCmessage[0] !== "0") { // do this if pose was detected
            // builds pose object
            let pose = reconstructPose(new Int16Array(WebRTCmessage[0]), new Int16Array(WebRTCmessage[1]));
            // clears the output canvas
            canvasScope.project.clear();
    
            // projects the poses skeleton on the existing svg skeleton
            Skeleton.flipPose(pose);
            illustration.updateSkeleton(pose, null);
            // illustration.draw(canvasScope, videoWidth, videoHeight);
            if (guiState.debug.showIllustrationDebug) {
                illustration.debugDraw(canvasScope);
            }
    
            canvasScope.project.activeLayer.scale(
                canvasWidth / videoWidth,
                canvasHeight / videoHeight,
                new canvasScope.Point(0, 0));
    
            let faceData = WebRTCmessage[2];
            if (faceData !== "0") {
    
                let face = {
                    positions: reconstructFaceData(WebRTCmessage[2]),
                    faceInViewConfidence: WebRTCmessage[3],
                };
    
                illustration.updateSkeleton(pose, face);
                illustration.draw(canvasScope, videoWidth, videoHeight);
            }
    
            let beforeStamp = WebRTCmessage[4];
            let renderedStamp = new Date().getTime();
    
            totalLatency.innerText = `Total pipeline latency: ${renderedStamp - beforeStamp}ms`;
            renderLatency.innerText = `Render latency: ${renderedStamp - afterExtractionStamp}ms`
        }
        WebRTCmessage = [];
    }
}

// Called when we are not the caller (ie we are the receiver)
// and the data channel has been opened
function handleDataChannelCreated(event) {
    console.log('dataChannel opened');

    dataChannel = event.channel;
    dataChannel.onopen = handleDataChannelStatusChange;
    dataChannel.onclose = handleDataChannelStatusChange;
}


// Handle status changes on the local end of the data
// channel; this is the end doing the sending of data
// in this example.

function handleDataChannelStatusChange(event) {
    if (dataChannel) {
        console.log('dataChannel status: ' + dataChannel.readyState);

        var state = dataChannel.readyState;

        if (state === 'open') {
            connectButton.disabled = true;
            WebRTCmessage = [];
            dataChannel.onmessage = handleDataChannelReceiveMessage;
            let statsInterval = window.setInterval(getConnectionStats,
                1000);
            configureRender();
            startTimer();
            transmit();
        } else {
            document.getElementById('warningDataChannel').innerText =
                'DataChannel closed. Refresh to reconnect.';
        }
    }
}

// Close the connection, including data channels if it's open.

function disconnectPeers() {

    // Close the RTCDataChannel if it's open.

    dataChannel.close();

    // Close the RTCPeerConnection

    peerConnection.close();

    dataChannel = null;
    peerConnection = null;

}

function errorHandler(error) {
    console.log(error);
}

// Taken from http://stackoverflow.com/a/105074/515584
// Strictly speaking, it's not a real UUID, but it gets the job done here
function createUUID() {
    function s4() {
        return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
    }

    return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() + s4() + s4();
}


/**
 * Converts a pose object to streamable array views, the corresponding
 * buffers are streamed
 *
 */
function deconstructPose(pose) {
    if (pose == null) return null;

    let confidences = new Int16Array(18);
    let positions = new Int16Array(34);

    confidences[0] = 10000 * pose.score; // to reduce transmission size
    for (let i = 0; i < pose.keypoints.length; i++) {
        confidences[i + 1] = 10000 * pose.keypoints[i].score;
        positions[i * 2] = pose.keypoints[i].position.x;
        positions[i * 2 + 1] = pose.keypoints[i].position.y;
    }

    return [confidences, positions];
}

/**
 * Converts streamed arrays (after view initialized) into a pose object for
 * animation rendering.
 *
 */
function reconstructPose(confidences, positions) {

    let pose = {
        'score': confidences[0] / 10000,
        'keypoints': [],
    };
    for (let i = 0; i < 17; i += 1) {
        pose.keypoints.push({
            'score': confidences[i + 1] / 10000,
            'part': parts[i],
            'position': {
                'x': positions[i * 2],
                'y': positions[i * 2 + 1],
            },
        });
    }
    return pose;
}


/**
 * Loads a the camera to be used in the demo
 *
 */
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById('video');
    video.width = videoWidth;
    video.height = videoHeight;

    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            width: videoWidth,
            height: videoHeight,
        },
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();

    return video;
}

const defaultPoseNetArchitecture = 'MobileNetV1';
const defaultQuantBytes = 2;
const defaultMultiplier = 1.0;
const defaultStride = 16;
const defaultInputResolution = 200;

const guiState = {
    avatarSVG: Object.keys(avatarSvgs)[0],
    debug: {
        showDetectionDebug: true,
        showIllustrationDebug: false,
        doNotTransmit: false,
    },
};

/**
 * Sets up dat.gui controller on the top-right of the window
 *
 */
function setupGui(cameras) {

    if (cameras.length > 0) {
        guiState.camera = cameras[0].deviceId;
    }

    const gui = new dat.GUI({width: 300});

    let multi = gui.addFolder('Image');
    gui.add(guiState, 'avatarSVG', Object.keys(avatarSvgs)).onChange(() => parseSVG(avatarSvgs[guiState.avatarSVG]));
    multi.open();

    let output = gui.addFolder('Debug control');
    output.add(guiState.debug, 'showDetectionDebug');
    output.add(guiState.debug, 'showIllustrationDebug');
    output.add(guiState.debug, 'doNotTransmit');
    output.open();
}

/**
 * Sets up a frames per second panel on the top-left of the window
 *
 */
function setupFPS() {
    stats.showPanel(0);
    document.getElementById('main').appendChild(stats.dom);
}

// more render configuration
function setupCanvas() {
    mobile = isMobile();
    if (mobile) {
        canvasWidth = Math.min(window.innerWidth, window.innerHeight);
        canvasHeight = canvasWidth;
        videoWidth *= 0.7;
        videoHeight *= 0.7;
    }

    canvasScope = paper.default;
    let canvas = document.querySelector('.illustration-canvas');
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    canvasScope.setup(canvas);
}

/**
 * Kicks off the demo by loading the posenet model, finding and loading
 * available camera devices, and setting off pose transmission device.
 */
export async function bindPage() {
    setupCanvas();

    toggleLoadingUI(true);
    setStatusText('Loading PoseNet model...');
    posenet = await posenet_module.load({
        architecture: defaultPoseNetArchitecture,
        outputStride: defaultStride,
        inputResolution: defaultInputResolution,
        multiplier: defaultMultiplier,
        quantBytes: defaultQuantBytes,
    });
    setStatusText('Loading FaceMesh model...');
    facemesh = await facemesh_module.load();

    setStatusText('Loading Avatar file...');
    let t0 = new Date();
    await parseSVG(Object.values(avatarSvgs)[0]);

    setStatusText('Setting up camera...');
    try {
        video = await loadVideo();
    } catch (e) {
        let info = document.getElementById('info');
        info.textContent = 'this device type is not supported yet, ' +
            'or this browser does not support video capture: ' + e.toString();
        info.style.display = 'block';
        throw e;
    }

    setupGui([], posenet);
    setupFPS();

    toggleLoadingUI(false);
}

// initiates svg skeleton to be used
navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
FileUtils.setDragDropHandler((result) => {
    parseSVG(result);
});

async function parseSVG(target) {
    let svgScope = await SVGUtils.importSVG(target /* SVG string or file path */);
    let skeleton = new Skeleton(svgScope);
    illustration = new PoseIllustration(canvasScope);
    illustration.bindSkeleton(skeleton, svgScope);
}

/**
 * Monitors inbound byte stream for the calculation of network transmission rate
 *
 */
function getConnectionStats() {

    let taken = [];
    peerConnection.getStats(null).then(stats => {
        let statsOutput = '';

        stats.forEach(report => {
            if (!report.id.startsWith('RTCDataChannel_')) return;
            Object.keys(report).forEach(statName => {
                if (monitors.includes(statName)) {

                    let bytesIntegral = parseInt(report[statName]);


                    if (bytesIntegral !== 0 && !taken.includes(statName)) {
                        let currentTime = new Date().getTime();
                        let timeIntegral = (currentTime - previousTime) / 1000;

                        let kbytesPerSecond = (bytesIntegral - previousBytesIntegral) / timeIntegral / 1000;
                        previousBytesIntegral = bytesIntegral;
                        previousTime = currentTime;
                        if (statName === 'bytesReceived') {
                            statsOutput += `<strong>kilobit rate: </strong> ${(kbytesPerSecond * 8).toFixed(2)} kb/s <br>`;
                            taken.push(statName);
                        } else {
                            statsOutput += `<strong>${statName}:</strong> ${kbytesPerSecond * 8} kb/s <br>`;
                            taken.push(statName);
                        }
                    }
                }
            });
        });
        document.querySelector('#bitstream-box').innerHTML = statsOutput;
    });
    return 0;
}

function startTimer() {
    previousTime = new Date().getTime();
}

/**
 * Sets up local and receiving renderers
 */
function configureRender() {
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    keypointCanvas.width = videoWidth;
    keypointCanvas.height = videoHeight;
}

// close websocket connection when tab closes
window.addEventListener('beforeunload', function(e) {
    disconnectPeers();
});

startup();
bindPage();
