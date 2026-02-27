// --- Bot Personality Presets ---
// Each bot uses the same steering behaviors but with different parameter sets.
// The parameters create visually distinct, character-driven movement.
export const CONFUSED_BOT = {
    name: "confused",
    color: "#F87171", // red-400
    label: "Confused",
    maxSpeed: 2.5,
    maxForce: 0.15,
    slowRadius: 30,
    arriveEasing: 0.5, // abrupt
    wanderWeight: 0.3,
    wanderRadius: 30,
    wanderDistance: 50,
    wanderJitter: 0.8,
    targetSwitchProbability: 0.004, // occasionally abandons targets (~20%/sec)
    revisitProbability: 0.12, // sometimes goes back to visited
    targetSelectionOrder: "random",
    speedVariance: 0.25,
    pauseDuration: 15,
    pauseVariance: 15, // moderate variance — sometimes lingers
    anticipationStrength: 0.1,
    followThroughStrength: 0.05,
    separationWeight: 0.3,
    separationRadius: 40,
    waypointArrivalRadius: 20,
};
export const MARKING_BOT = {
    name: "marking",
    color: "#34D399", // emerald-400
    label: "Marking",
    maxSpeed: 2.0,
    maxForce: 0.12,
    slowRadius: 50,
    arriveEasing: 1.8, // gentle deceleration
    wanderWeight: 0,
    wanderRadius: 0,
    wanderDistance: 0,
    wanderJitter: 0,
    targetSwitchProbability: 0,
    revisitProbability: 0,
    targetSelectionOrder: "nearest",
    speedVariance: 0.05,
    pauseDuration: 30,
    pauseVariance: 5,
    anticipationStrength: 0.3,
    followThroughStrength: 0.2,
    separationWeight: 0.5,
    separationRadius: 50,
    waypointArrivalRadius: 12,
};
export const ORGANIZING_BOT = {
    name: "organizing",
    color: "#60A5FA", // blue-400
    label: "Organizing",
    maxSpeed: 3.2,
    maxForce: 0.2,
    slowRadius: 35,
    arriveEasing: 1.2,
    wanderWeight: 0,
    wanderRadius: 0,
    wanderDistance: 0,
    wanderJitter: 0,
    targetSwitchProbability: 0,
    revisitProbability: 0,
    targetSelectionOrder: "nearest",
    speedVariance: 0.08,
    pauseDuration: 10,
    pauseVariance: 3,
    anticipationStrength: 0.15,
    followThroughStrength: 0.1,
    separationWeight: 0.4,
    separationRadius: 45,
    waypointArrivalRadius: 10,
};
export const GRID_BOT = {
    name: "grid",
    color: "#A78BFA", // violet-400
    label: "Grid",
    maxSpeed: 2.0,
    maxForce: 0.12,
    slowRadius: 60,
    arriveEasing: 2.2, // very gentle, anticipatory deceleration
    wanderWeight: 0,
    wanderRadius: 0,
    wanderDistance: 0,
    wanderJitter: 0,
    targetSwitchProbability: 0,
    revisitProbability: 0,
    targetSelectionOrder: "nearest",
    speedVariance: 0.03,
    pauseDuration: 25,
    pauseVariance: 5,
    anticipationStrength: 0.5,
    followThroughStrength: 0.4,
    separationWeight: 0.5,
    separationRadius: 55,
    waypointArrivalRadius: 10,
};
export const UNCONVENTIONAL_BOT = {
    name: "unconventional",
    color: "#FBBF24", // amber-400
    label: "Unconventional",
    maxSpeed: 2.2,
    maxForce: 0.14,
    slowRadius: 45,
    arriveEasing: 1.6,
    wanderWeight: 0.05, // tiny bit of personality drift
    wanderRadius: 15,
    wanderDistance: 40,
    wanderJitter: 0.4,
    targetSwitchProbability: 0,
    revisitProbability: 0,
    targetSelectionOrder: "center-out", // starts from middle, works outward
    speedVariance: 0.1,
    pauseDuration: 20,
    pauseVariance: 8,
    anticipationStrength: 0.25,
    followThroughStrength: 0.2,
    separationWeight: 0.4,
    separationRadius: 45,
    waypointArrivalRadius: 12,
};
export const ALL_PERSONALITIES = [
    CONFUSED_BOT,
    MARKING_BOT,
    ORGANIZING_BOT,
    GRID_BOT,
    UNCONVENTIONAL_BOT,
];
