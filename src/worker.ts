import Scene from "./Scene";

const scene = new Scene();

self.onmessage = ({ data }) =>
{
    switch (data.action)
    {
        case "Transfer::WebGPU":
            return scene.create(data.canvas).then(() => setTimeout(() =>
                scene.resize(data.width, data.height)
            , 50 / 3));

        case "Transfer::2D":
            return scene.output = data.canvas;

        case "Resize::Window":
            return scene.resize(data.width, data.height);
    }

};

self.onerror = console.error;

export default self;
