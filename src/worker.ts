import Scene from "./Scene";

const scene = new Scene();

self.onmessage = async ({ data }) =>
{
    switch (data.action)
    {
        case "Transfer::WebGPU":
            await scene.create(data.canvas);
            self.postMessage({ action: "Resize::Window" });
        break;

        case "Transfer::2D":
            return scene.output = data.canvas;

        case "Resize::Window":
            return scene.resize(data.width, data.height);
    }

};

self.onerror = console.error;

export default self;
