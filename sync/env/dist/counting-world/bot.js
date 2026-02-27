// --- Bot Entity ---
// A bot navigates waypoints using steering behaviors parameterized by personality.
// Update loop: evaluate behaviors → combine forces → update velocity → update position.
import { vec2, add, scale, length, normalize, limit, dist, arrive, wander, separation, combineForces, } from "./steering.js";
export function createBot(id, personality, startPosition, waypoints) {
    return {
        id,
        personality,
        position: { ...startPosition },
        velocity: vec2(0, 0),
        acceleration: vec2(0, 0),
        waypoints,
        currentTargetIndex: selectInitialTarget(personality, waypoints),
        visitedIndices: new Set(),
        allVisited: false,
        wanderState: { angle: Math.random() * Math.PI * 2 },
        pauseTimer: 0,
        currentSpeedMultiplier: 1,
        speedMultiplierTarget: 1,
        trail: [],
        trailMaxLength: 120,
        previousDirection: vec2(0, 0),
        onArrival: null,
        countTally: 0,
        dynamicTarget: null,
    };
}
function selectInitialTarget(personality, waypoints) {
    if (waypoints.length === 0)
        return 0;
    switch (personality.targetSelectionOrder) {
        case "center-out": {
            // Start from the waypoint closest to the centroid of all waypoints
            let cx = 0, cy = 0;
            for (const wp of waypoints) {
                cx += wp.x;
                cy += wp.y;
            }
            cx /= waypoints.length;
            cy /= waypoints.length;
            let bestIdx = 0, bestDist = Infinity;
            for (let i = 0; i < waypoints.length; i++) {
                const d = dist(vec2(cx, cy), waypoints[i]);
                if (d < bestDist) {
                    bestDist = d;
                    bestIdx = i;
                }
            }
            return bestIdx;
        }
        case "farthest-first":
            return waypoints.length - 1;
        default:
            return 0;
    }
}
function selectNextTarget(bot) {
    const { personality, waypoints, visitedIndices, currentTargetIndex, position } = bot;
    // Confused bot might revisit
    if (personality.revisitProbability > 0 && Math.random() < personality.revisitProbability && visitedIndices.size > 0) {
        const visited = Array.from(visitedIndices);
        return visited[Math.floor(Math.random() * visited.length)];
    }
    const unvisited = waypoints
        .map((_, i) => i)
        .filter((i) => !visitedIndices.has(i));
    if (unvisited.length === 0)
        return currentTargetIndex; // all visited
    switch (personality.targetSelectionOrder) {
        case "nearest": {
            let bestIdx = unvisited[0];
            let bestDist = dist(position, waypoints[bestIdx]);
            for (const i of unvisited) {
                const d = dist(position, waypoints[i]);
                if (d < bestDist) {
                    bestDist = d;
                    bestIdx = i;
                }
            }
            return bestIdx;
        }
        case "sequential":
            // Find next unvisited after current
            for (let offset = 1; offset <= waypoints.length; offset++) {
                const idx = (currentTargetIndex + offset) % waypoints.length;
                if (!visitedIndices.has(idx))
                    return idx;
            }
            return unvisited[0];
        case "random":
            return unvisited[Math.floor(Math.random() * unvisited.length)];
        case "farthest-first": {
            let bestIdx = unvisited[0];
            let bestDist = 0;
            for (const i of unvisited) {
                const d = dist(position, waypoints[i]);
                if (d > bestDist) {
                    bestDist = d;
                    bestIdx = i;
                }
            }
            return bestIdx;
        }
        case "center-out": {
            // Sort unvisited by distance from centroid (closest first, but we've already visited the closest)
            let cx = 0, cy = 0;
            for (const wp of waypoints) {
                cx += wp.x;
                cy += wp.y;
            }
            cx /= waypoints.length;
            cy /= waypoints.length;
            const centroid = vec2(cx, cy);
            const sorted = unvisited.slice().sort((a, b) => dist(centroid, waypoints[a]) - dist(centroid, waypoints[b]));
            return sorted[0];
        }
        default:
            return unvisited[0];
    }
}
export function updateBot(bot, otherBotPositions, dt) {
    const p = bot.personality;
    // Handle pause
    if (bot.pauseTimer > 0) {
        bot.pauseTimer -= 1;
        // Apply braking during pause
        bot.velocity = scale(bot.velocity, 0.85);
        updateTrail(bot);
        return;
    }
    // Check if at current target (skipped for strategy-driven bots)
    if (bot.waypoints.length > 0 && !bot.dynamicTarget) {
        const target = bot.waypoints[bot.currentTargetIndex];
        const d = dist(bot.position, target);
        if (d < p.waypointArrivalRadius) {
            // Fire arrival callback (blob interaction)
            let shouldMarkVisited = true;
            if (bot.onArrival) {
                shouldMarkVisited = bot.onArrival(bot, bot.currentTargetIndex);
            }
            bot.countTally++;
            if (shouldMarkVisited) {
                bot.visitedIndices.add(bot.currentTargetIndex);
            }
            // Check if all visited
            const unvisitedCount = bot.waypoints.length - bot.visitedIndices.size;
            if (unvisitedCount === 0) {
                bot.allVisited = true;
            }
            // Pause at waypoint
            const pauseVar = Math.round((Math.random() - 0.5) * 2 * p.pauseVariance);
            bot.pauseTimer = Math.max(0, p.pauseDuration + pauseVar);
            // Select next target (unless all done)
            if (!bot.allVisited) {
                bot.currentTargetIndex = selectNextTarget(bot);
            }
            return;
        }
        // Confused bot: random target switching mid-path
        if (p.targetSwitchProbability > 0 && Math.random() < p.targetSwitchProbability) {
            bot.currentTargetIndex = selectNextTarget(bot);
        }
    }
    // --- Compute steering forces ---
    const agent = {
        position: bot.position,
        velocity: bot.velocity,
        maxSpeed: p.maxSpeed * bot.currentSpeedMultiplier,
        maxForce: p.maxForce,
    };
    const forces = [];
    // Dynamic target takes priority (strategy-driven bots)
    if (bot.dynamicTarget) {
        const arriveForce = arrive(agent, bot.dynamicTarget, p.slowRadius, p.arriveEasing);
        forces.push({ force: arriveForce, weight: 1.0 });
    }
    else if (bot.waypoints.length > 0) {
        // Arrive at current waypoint target
        const target = bot.waypoints[bot.currentTargetIndex];
        const arriveForce = arrive(agent, target, p.slowRadius, p.arriveEasing);
        forces.push({ force: arriveForce, weight: 1.0 });
    }
    // Wander
    if (p.wanderWeight > 0) {
        const w = wander(agent, bot.wanderState, p.wanderRadius, p.wanderDistance, p.wanderJitter);
        forces.push({ force: w.force, weight: p.wanderWeight });
        bot.wanderState.angle = w.newAngle;
    }
    // Separation from other bots
    if (p.separationWeight > 0 && otherBotPositions.length > 0) {
        const sepForce = separation(agent, otherBotPositions, p.separationRadius);
        forces.push({ force: sepForce, weight: p.separationWeight });
    }
    // Combine forces
    const totalForce = combineForces(forces);
    bot.acceleration = limit(totalForce, p.maxForce);
    // --- Anticipation ---
    // Lean back slightly before direction changes
    if (p.anticipationStrength > 0 && length(bot.velocity) > 0.5) {
        const currentDir = normalize(bot.velocity);
        const accelDir = normalize(bot.acceleration);
        const dirChange = 1 - (currentDir.x * accelDir.x + currentDir.y * accelDir.y); // 0 = same dir, 2 = opposite
        if (dirChange > 0.3) {
            // Apply a slight counter-force (anticipation lean)
            const antiForce = scale(currentDir, -p.anticipationStrength * dirChange * p.maxForce);
            bot.acceleration = add(bot.acceleration, antiForce);
        }
    }
    // --- Update velocity ---
    bot.velocity = add(bot.velocity, scale(bot.acceleration, dt));
    bot.velocity = limit(bot.velocity, p.maxSpeed * bot.currentSpeedMultiplier);
    // --- Follow-through ---
    // Slight overshoot damping (higher = more overshoot allowed)
    const dampening = 1 - (0.02 * (1 - p.followThroughStrength));
    bot.velocity = scale(bot.velocity, dampening);
    // --- Speed variation ---
    // Slowly drift speed multiplier toward a new random target
    if (p.speedVariance > 0) {
        if (Math.random() < 0.02) {
            bot.speedMultiplierTarget = 1 + (Math.random() - 0.5) * 2 * p.speedVariance;
        }
        bot.currentSpeedMultiplier += (bot.speedMultiplierTarget - bot.currentSpeedMultiplier) * 0.05;
    }
    // --- Update position ---
    bot.position = add(bot.position, scale(bot.velocity, dt));
    // Store direction for next frame's anticipation
    if (length(bot.velocity) > 0.1) {
        bot.previousDirection = normalize(bot.velocity);
    }
    updateTrail(bot);
}
function updateTrail(bot) {
    bot.trail.push({ ...bot.position });
    if (bot.trail.length > bot.trailMaxLength) {
        bot.trail.shift();
    }
}
/**
 * Reset a bot: clear visited, reposition, select new initial target.
 */
export function resetBot(bot, startPosition) {
    bot.position = { ...startPosition };
    bot.velocity = vec2(0, 0);
    bot.acceleration = vec2(0, 0);
    bot.visitedIndices.clear();
    bot.allVisited = false;
    bot.pauseTimer = 0;
    bot.trail = [];
    bot.countTally = 0;
    bot.dynamicTarget = null;
    bot.currentTargetIndex = selectInitialTarget(bot.personality, bot.waypoints);
}
