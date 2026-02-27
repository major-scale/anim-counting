// --- Steering Behaviors Engine ---
// Craig Reynolds-style steering behaviors. Each returns a force vector.
// Combined via weighted sum to produce organic, lifelike bot movement.
export function vec2(x, y) {
    return { x, y };
}
export function add(a, b) {
    return { x: a.x + b.x, y: a.y + b.y };
}
export function sub(a, b) {
    return { x: a.x - b.x, y: a.y - b.y };
}
export function scale(v, s) {
    return { x: v.x * s, y: v.y * s };
}
export function length(v) {
    return Math.sqrt(v.x * v.x + v.y * v.y);
}
export function normalize(v) {
    const len = length(v);
    if (len === 0)
        return { x: 0, y: 0 };
    return { x: v.x / len, y: v.y / len };
}
export function limit(v, max) {
    const len = length(v);
    if (len <= max)
        return v;
    return scale(normalize(v), max);
}
export function dist(a, b) {
    return length(sub(a, b));
}
export function dot(a, b) {
    return a.x * b.x + a.y * b.y;
}
export function rotate(v, angle) {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return { x: v.x * cos - v.y * sin, y: v.x * sin + v.y * cos };
}
// --- Steering Behaviors ---
/**
 * Seek: steer toward target at max speed.
 */
export function seek(agent, target) {
    const desired = scale(normalize(sub(target, agent.position)), agent.maxSpeed);
    return limit(sub(desired, agent.velocity), agent.maxForce);
}
/**
 * Flee: steer away from target at max speed.
 */
export function flee(agent, target) {
    const desired = scale(normalize(sub(agent.position, target)), agent.maxSpeed);
    return limit(sub(desired, agent.velocity), agent.maxForce);
}
/**
 * Arrive: seek but decelerate smoothly within slowRadius.
 * easing controls the deceleration curve (1 = linear, <1 = ease-in, >1 = ease-out).
 */
export function arrive(agent, target, slowRadius, easing = 1) {
    const offset = sub(target, agent.position);
    const d = length(offset);
    if (d < 0.5)
        return scale(agent.velocity, -1); // brake to stop
    let speed;
    if (d < slowRadius) {
        const t = d / slowRadius; // 0..1
        const eased = Math.pow(t, easing);
        speed = agent.maxSpeed * eased;
    }
    else {
        speed = agent.maxSpeed;
    }
    const desired = scale(normalize(offset), speed);
    return limit(sub(desired, agent.velocity), agent.maxForce);
}
export function wander(agent, state, wanderRadius, wanderDistance, wanderJitter) {
    // Jitter the wander angle
    const newAngle = state.angle + (Math.random() - 0.5) * wanderJitter;
    // Circle center is projected ahead of agent
    const vel = length(agent.velocity) > 0.01 ? normalize(agent.velocity) : vec2(1, 0);
    const circleCenter = add(agent.position, scale(vel, wanderDistance));
    // Target point on circle
    const target = add(circleCenter, vec2(Math.cos(newAngle) * wanderRadius, Math.sin(newAngle) * wanderRadius));
    const desired = scale(normalize(sub(target, agent.position)), agent.maxSpeed);
    const force = limit(sub(desired, agent.velocity), agent.maxForce);
    return { force, newAngle };
}
/**
 * Path Following: follow a sequence of waypoints with smooth interpolation.
 * Returns the steering force and current waypoint index.
 */
export function pathFollow(agent, waypoints, currentIndex, arrivalRadius, slowRadius, easing = 1) {
    if (waypoints.length === 0)
        return { force: vec2(0, 0), nextIndex: 0 };
    const target = waypoints[currentIndex];
    const d = dist(agent.position, target);
    // Advance to next waypoint if close enough
    let nextIndex = currentIndex;
    if (d < arrivalRadius && currentIndex < waypoints.length - 1) {
        nextIndex = currentIndex + 1;
    }
    // Use arrive for current target
    const force = arrive(agent, waypoints[nextIndex], slowRadius, easing);
    return { force, nextIndex };
}
export function obstacleAvoidance(agent, obstacles, feelerLength) {
    const vel = length(agent.velocity) > 0.01 ? normalize(agent.velocity) : vec2(1, 0);
    const ahead = add(agent.position, scale(vel, feelerLength));
    const halfAhead = add(agent.position, scale(vel, feelerLength * 0.5));
    let nearest = null;
    let nearestDist = Infinity;
    for (const obs of obstacles) {
        // Check if feeler intersects obstacle (simple circle check at two points)
        const d1 = dist(ahead, obs.position);
        const d2 = dist(halfAhead, obs.position);
        const d3 = dist(agent.position, obs.position);
        const minD = Math.min(d1, d2, d3);
        if (minD < obs.radius && minD < nearestDist) {
            nearest = obs;
            nearestDist = minD;
        }
    }
    if (!nearest)
        return vec2(0, 0);
    const avoidForce = sub(ahead, nearest.position);
    return limit(scale(normalize(avoidForce), agent.maxForce), agent.maxForce);
}
/**
 * Separation: steer away from nearby neighbors to maintain spacing.
 */
export function separation(agent, neighbors, desiredSeparation) {
    let steer = vec2(0, 0);
    let count = 0;
    for (const other of neighbors) {
        const d = dist(agent.position, other);
        if (d > 0 && d < desiredSeparation) {
            const diff = normalize(sub(agent.position, other));
            steer = add(steer, scale(diff, 1 / d)); // weight by inverse distance
            count++;
        }
    }
    if (count > 0) {
        steer = scale(steer, 1 / count);
        steer = scale(normalize(steer), agent.maxSpeed);
        steer = sub(steer, agent.velocity);
        steer = limit(steer, agent.maxForce);
    }
    return steer;
}
export function combineForces(forces) {
    let result = vec2(0, 0);
    for (const { force, weight } of forces) {
        result = add(result, scale(force, weight));
    }
    return result;
}
