override DIMENSION_SIZE: u32 = 16;

@group(0) @binding(0) var<uniform> resolution: vec3f;
@group(0) @binding(1) var<storage, read_write> Values: array<vec3u>;

@compute @workgroup_size(DIMENSION_SIZE, DIMENSION_SIZE)
fn mainCompute(@builtin(global_invocation_id) globalInvocation: vec3u)
{
    let coord = vec2f(globalInvocation.xy);

    if (all(coord < resolution.xy))
    {
        let index = u32(coord.x + (resolution.y - coord.y) * resolution.x);
        Values[index] = vec3u(255);
    }
}
