self.onmessage = ({ data }) => import('./Scene').then(
    Scene => Scene.create(data)
);

self.onerror = console.error;

export default self;
