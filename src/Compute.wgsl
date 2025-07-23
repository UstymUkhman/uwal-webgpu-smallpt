const EPS = 1e-4;
alias Refl_t = u32;

const DIFF: Refl_t = 0;
const SPEC: Refl_t = 1;
const REFR: Refl_t = 2;

var<private> rnd: vec3u;
override SAMPS = 200u / 4;
const GAMMA = vec3f(1 / 2.2);

override DIMENSION_SIZE = 16u;
var<private> INV_SAMPS = 1 / SAMPS;
const INF = 3.40282346638528859812e+38f;

struct Sphere
{
    p: vec3f, rad: f32,
    e: vec3f, refl: Refl_t, c: vec3f
};

struct Ray { o: vec3f, d: vec3f };

fn init_rnd(id: vec3u, seed: vec3u)
{
    const A = vec3(
        1741651 * 1009,
        140893 * 1609 * 13,
        6521 * 983 * 7 * 2
    );

    rnd = (id * A) ^ seed;
}

fn rand() -> f32
{
    const C = vec3(
        60493 * 9377,
        11279 * 2539 * 23,
        7919 * 631 * 5 * 3
    );

    rnd = (rnd * C) ^ (rnd.yzx >> vec3(4u));
    return f32(rnd.x ^ rnd.y) / f32(0xffffffff);
}

fn intersect_sphere(s: Sphere, r: Ray) -> f32
{
    let op = s.p - r.o;
    let b = dot(op, r.d);

    var det = b * b - dot(op, op) + s.rad * s.rad;

    if (det < 0) { return 0; }
    else { det = sqrt(det); }

    var t = b - det;

    if (t > EPS) { return t; }
    else
    {
        t = b + det;
        return select(0, t, t > EPS);
    }
}

fn intersect(r: Ray, t: ptr<function, f32>, id: ptr<function, u32>) -> bool
{
    *t = INF;

    for (var i = SPHERES; i > 0; i--)
    {
        let d = intersect_sphere(spheres[i], r);

        if (d > 0 && d < *t)
        {
            *t = d;
            *id = i;
        }
    }

    return *t < INF;
}

@group(0) @binding(0) var<uniform> Xi: vec3u;
@group(0) @binding(1) var<uniform> res: vec3f;
@group(0) @binding(2) var<storage, read_write> color: array<vec4u>;
@group(0) @binding(3) var<storage, read> spheres: array<Sphere, SPHERES>;

@compute @workgroup_size(DIMENSION_SIZE, DIMENSION_SIZE)
fn compute(@builtin(global_invocation_id) globalInvocation: vec3u)
{
    let coord = vec2f(globalInvocation.xy);
    init_rnd(globalInvocation, Xi);

    if (all(coord < res.xy))
    {
        let cam = Ray(vec3f(50, 52, 295.6), normalize(vec3f(0, -0.042612, -1)));
        let i = u32(coord.x + coord.y * res.x);

        let x = coord.x / res.x;
        let y = coord.y / res.y;

        let cx = vec3f(res.x * 0.5135 / res.y);
        let cy = normalize(cross(cx, cam.d)) * 0.5135;

        var c = vec3f(vec3f(color[i].rgb) / 255);

        for (var sy = 0u; sy < 2; sy++)
        {
            for (var sx = 0u; sx < 2; sx++)
            {
                var r = vec3f(0);

                for (var s = 0u; s < SAMPS; s++)
                {
                    let r1 = rand() * 2;
                    let r2 = rand() * 2;

                    let dx = select(1 - sqrt(2 - r1), sqrt(r1) - 1, r1 < 1);
                    let dy = select(1 - sqrt(2 - r2), sqrt(r2) - 1, r2 < 1);

                    let d = cx * (((f32(sx) + 0.5 + dx) / 2 + x) / res.x - 0.5) +
                            cy * (((f32(sy) + 0.5 + dy) / 2 + y) / res.y - 0.5) + cam.d;

                    // r += radiance(Ray(cam.o + d * 140, normalize(d)), 0, Xi) * INV_SAMPS;
                }

                c += clamp(r, vec3f(0), vec3f(1)) * 0.25;
            }
        }

        color[i] = vec4u(vec3u(pow(clamp(c, vec3f(0), vec3f(1)), GAMMA) * 255 + 0.5), 255);
    }
}
