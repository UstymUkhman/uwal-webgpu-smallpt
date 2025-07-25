const EPS = 1e-4;
const INF = 1e20f;
alias Refl_t = u32;

const DIFF: Refl_t = 0;
const SPEC: Refl_t = 1;
const REFR: Refl_t = 2;

const PI = radians(180);
var<private> rnd: vec3u;

const GAMMA = vec3f(1 / 2.2);
override DIMENSION_SIZE = 16u;

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

    for (var s = 0u; s < SPHERES; s++)
    {
        let d = intersect_sphere(spheres[s], r);

        if (d != 0f && d < *t)
        {
            *t = d;
            *id = s;
        }
    }

    return *t < INF;
}

fn radiance(ray: Ray, depth: u32) -> vec3f
{
    var t: f32;
    var id = 0u;

    var r = ray;
    var d = depth;

    var cl = vec3f(0);
    var cf = vec3f(1);

    while (true)
    {
        if (!intersect(r, &t, &id))
        { return cl; }

        let obj = spheres[id];

        let x = r.o + r.d * t;
        let n = normalize(x - obj.p);
        let nl = select(-n, n, dot(n, r.d) < 0);
        var f = obj.c;

        let p = select(
            select(f.z, f.y, f.y > f.z),
            f.x, f.x > f.y && f.x > f.z
        );

        cl += cf * obj.e;
        d++;

        if (d > 5)
        {
            if (rand() < p) { f *= (1 / p); }
            else { return cl; }
        }

        cf *= f;

        if (obj.refl == DIFF)
        {
            let r1 = 2 * PI * rand();
            let r2 = rand();
            let r2s = sqrt(r2);

            let w = nl;
            let u = normalize(cross(select(vec3f(1), vec3f(0, 1, 0), abs(w.x) > 0.1), w));

            let v = cross(w, u);
            let d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2));

            r = Ray(x, d);
            continue;
        }
        else if (obj.refl == SPEC)
        {
            r = Ray(x, r.d - n * 2 * dot(n, r.d));
            continue;
        }

        let reflRay = Ray(x, r.d - n * 2 * dot(n, r.d));
        let into = dot(n, nl) > 0;
        let nc = 1f;
        let nt = 1.5;
        let nnt = select(nt / nc, nc / nt, into);
        let ddn = dot(r.d, nl);
        let cos2t = 1 - nnt * nnt * (1 - ddn * ddn);

        if (cos2t < 0)
        {
            r = reflRay;
            continue;
        }

        let tdir = normalize(r.d * nnt - n * select(-1f, 1f, into) * (ddn * nnt + sqrt(cos2t)));

        let a = nt - nc;
        let b = nt + nc;
        let R0 = a * a / (b * b);
        let c = 1 - select(dot(tdir, n), -ddn, into);

        let Re = R0 + (1 - R0) * c * c * c * c * c;
        let Tr = 1 - Re;
        let P = 0.25 + 0.5 * Re;
        let RP = Re / P;
        let TP = Tr / (1 - P);

        if (rand() < P)
        {
            cf *= RP;
            r = reflRay;
        }
        else
        {
            cf *= TP;
            r = Ray(x, tdir);
        }

        continue;
    }

    return cl;
}

@group(0) @binding(0) var<uniform> res: vec3f;
@group(0) @binding(1) var<uniform> seed: vec3u;
@group(0) @binding(2) var<storage, read_write> color: array<vec4u>;
@group(0) @binding(3) var<storage, read> spheres: array<Sphere, SPHERES>;

@compute @workgroup_size(DIMENSION_SIZE, DIMENSION_SIZE)
fn compute(@builtin(global_invocation_id) globalInvocation: vec3u)
{
    let coord = vec2f(globalInvocation.xy);
    init_rnd(globalInvocation, seed);

    if (all(coord < res.xy))
    {
        let x = coord.x / res.x;
        let y = coord.y / res.y;

        let cam = Ray(
            vec3f(50, 52, 295.6),
            normalize(vec3f(0, -0.042612, -1))
        );

        let i = u32(coord.x + (res.y - coord.y) * res.x);
        var c = vec3f(vec3f(color[i].rgb) / 255);

        let cx = vec3f(res.x * 0.5135 / res.y);
        let cy = normalize(cross(cx, cam.d)) * 0.5135;

        for (var sy = 0u; sy < 2; sy++)
        {
            for (var sx = 0u; sx < 2; sx++)
            {
                let r1 = rand() * 2;
                let r2 = rand() * 2;

                let dx = select(1 - sqrt(2 - r1), sqrt(r1) - 1, r1 < 1);
                let dy = select(1 - sqrt(2 - r2), sqrt(r2) - 1, r2 < 1);

                var d = cx * (((f32(sx) + 0.5 + dx) / 2 + x) / res.x - 0.5) +
                        cy * (((f32(sy) + 0.5 + dy) / 2 + y) / res.y - 0.5) + cam.d;

                let ray = radiance(Ray(cam.o + d * 140, normalize(d)), 0);
                c += clamp(ray, vec3f(0), vec3f(1)) * 0.25;
            }
        }

        color[i] = vec4u(vec3u(pow(clamp(c, vec3f(0), vec3f(1)), GAMMA) * 255 + 0.5), 255);
    }
}
