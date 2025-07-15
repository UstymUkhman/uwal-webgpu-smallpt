@group(0) @binding(1) var<storage, read_write> values: array<vec4u>;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f
{
    return vec4f(GetQuadCoord(index), 0, 1);
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f
{
    let pos = vec2u(position.xy);
    let res = vec2u(resolution.xy);

    let x = pos.x % res.x;
    let y = pos.y % res.y;
    let i = x + y * res.x;

    return vec4f(values[i] / 255);
}
