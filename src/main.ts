import Worker from './worker?worker';

const canvas = scene.transferControlToOffscreen();
(new Worker()).postMessage(canvas, [canvas]);
