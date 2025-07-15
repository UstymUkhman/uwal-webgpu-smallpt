override DIMENSION_SIZE: u32 = 16;

@group(0) @binding(0) var<uniform> resolution: vec3f;
@group(0) @binding(1) var<storage, read_write> values: array<vec4u>;

@compute @workgroup_size(DIMENSION_SIZE, DIMENSION_SIZE)
fn compute(@builtin(global_invocation_id) globalInvocation: vec3u)
{
    let coord = vec2f(globalInvocation.xy);
    let center = vec2f(resolution.xy) / 2;

    let dist = distance(coord, center);
    let white = dist / 32.0 % 2.0 < 1;

    if (all(coord < resolution.xy))
    {
        let index = u32(coord.x + coord.y * resolution.x);
        values[index] = select(vec4u(255, 0, 0, 255), vec4u(255), white);
    }
}
