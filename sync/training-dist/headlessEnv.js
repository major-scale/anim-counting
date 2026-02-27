// --- Headless Counting World Environment ---
// Runs the same simulation logic as the browser version, without rendering.
// All simulation state (marks, visits, counting progress) is tracked.
// Produces observation vectors for the RL agent.
import { ALL_PERSONALITIES, } from "./counting-world/botPersonalities.js";
import { createBot, updateBot, } from "./counting-world/bot.js";
import { vec2 } from "./counting-world/steering.js";
import { countBlob, markBlob, updateBlobAnimations, updateCarriedBlobs, } from "./counting-world/blob.js";
import { createStrategy, tickStrategy, computeLinePlacement, computeGridPlacement, } from "./counting-world/countingStrategy.js";
import { randomArrangement, generateArrangement, } from "./arrangements.js";
// --- Constants ---
const WORLD_WIDTH = 1400;
const WORLD_HEIGHT = 1000;
const MARGIN = 120;
const MIN_SEPARATION = 50;
const DT = 1;
const MAX_BLOB_COUNT = 25; // observation vector padding
const BOUNDARY_MARGIN = 40;
export const DEFAULT_CONFIG = {
    stage: 1,
    conservation: true,
    blobCountMin: 6,
    blobCountMax: 25,
    maxSteps: 5000,
};
// --- Observation vector ---
// [bot_x, bot_y, bot_state, blob_0_x, blob_0_y, ..., blob_24_x, blob_24_y,
//  blob_0_marked, ..., blob_24_marked, actual_blob_count, phase_indicator]
// Total: 2 + 1 + 50 + 25 + 1 + 1 = 80
export const OBS_SIZE = 80;
function botStateValue(bot) {
    if (bot.allVisited)
        return 2; // done
    if (bot.pauseTimer > 0)
        return 1; // counting/paused at blob
    return 0; // seeking
}
export function getObservation(state) {
    const obs = new Array(OBS_SIZE).fill(0);
    const { bot, blobField, phase } = state;
    // Normalize positions to [0, 1]
    obs[0] = bot.position.x / WORLD_WIDTH;
    obs[1] = bot.position.y / WORLD_HEIGHT;
    obs[2] = botStateValue(bot);
    // Blob positions (padded to MAX_BLOB_COUNT)
    const blobOffset = 3;
    for (let i = 0; i < blobField.blobs.length && i < MAX_BLOB_COUNT; i++) {
        obs[blobOffset + i * 2] = blobField.blobs[i].position.x / WORLD_WIDTH;
        obs[blobOffset + i * 2 + 1] = blobField.blobs[i].position.y / WORLD_HEIGHT;
    }
    // Blob marked status (1 if marked by the active bot, 0 otherwise)
    const markOffset = blobOffset + MAX_BLOB_COUNT * 2; // 3 + 50 = 53
    for (let i = 0; i < blobField.blobs.length && i < MAX_BLOB_COUNT; i++) {
        const blob = blobField.blobs[i];
        obs[markOffset + i] = blob.countedBy.has(bot.id) ? 1 : 0;
    }
    // Actual blob count this episode
    obs[markOffset + MAX_BLOB_COUNT] = blobField.blobs.length; // index 78
    // Phase indicator
    obs[markOffset + MAX_BLOB_COUNT + 1] = phase === "predict" ? 1 : 0; // index 79
    return obs;
}
// --- Bot selection by stage ---
function selectBotPersonality(stage) {
    const byName = (name) => ALL_PERSONALITIES.find((p) => p.name === name);
    switch (stage) {
        case 1:
            return byName("marking");
        case 2:
            return byName("confused");
        case 3:
            return Math.random() < 0.5 ? byName("marking") : byName("confused");
        case 4: {
            const pool = ["marking", "confused", "organizing", "grid"];
            return byName(pool[Math.floor(Math.random() * pool.length)]);
        }
        case 5:
            return ALL_PERSONALITIES[Math.floor(Math.random() * ALL_PERSONALITIES.length)];
    }
}
// --- Arrival handlers (same logic as CountingWorldDemo) ---
function createArrivalHandler(personalityName, field) {
    switch (personalityName) {
        case "confused": {
            const stopAfter = field.totalCount + 3 + Math.floor(Math.random() * 4);
            return (bot, idx) => {
                const blob = field.blobs[idx];
                if (!blob)
                    return true; // blob removed (conservation OFF)
                countBlob(blob, bot.id, bot.personality.color);
                const forgets = Math.random() < 0.15;
                if (bot.countTally >= stopAfter) {
                    for (let i = 0; i < bot.waypoints.length; i++) {
                        bot.visitedIndices.add(i);
                    }
                    return true;
                }
                return !forgets;
            };
        }
        case "marking":
            return (bot, idx) => {
                const blob = field.blobs[idx];
                if (!blob)
                    return true; // blob removed (conservation OFF)
                const isNew = countBlob(blob, bot.id, bot.personality.color);
                if (isNew) {
                    markBlob(blob, bot.id);
                }
                return true;
            };
        case "organizing":
        case "grid":
            return (bot, idx) => {
                const blob = field.blobs[idx];
                if (!blob)
                    return true; // blob removed (conservation OFF)
                countBlob(blob, bot.id, bot.personality.color);
                return true;
            };
        case "unconventional":
            return (bot, idx) => {
                const blob = field.blobs[idx];
                if (!blob)
                    return true; // blob removed (conservation OFF)
                countBlob(blob, bot.id, bot.personality.color);
                return true;
            };
        default:
            return () => true;
    }
}
// --- Blob field creation from arrangement positions ---
function createBlobFieldFromPositions(positions) {
    const BLOB_COLORS = [
        "#F9A8D4", "#FCA5A5", "#FDBA74", "#FCD34D",
        "#86EFAC", "#67E8F9", "#A5B4FC", "#C4B5FD",
    ];
    const blobs = positions.map((pos, i) => ({
        id: i,
        position: { ...pos },
        radius: 10 + Math.random() * 8,
        baseColor: BLOB_COLORS[i % BLOB_COLORS.length],
        countedBy: new Set(),
        markedBy: null,
        isCarried: false,
        carriedBy: null,
        placedPosition: null,
        markGlow: 0,
        countFlash: 0,
        countFlashColor: "#ffffff",
    }));
    return { blobs, totalCount: blobs.length };
}
// --- Conservation OFF: blob mutation ---
function mutateBlobs(field) {
    // 5% chance per step of adding or removing one blob
    if (Math.random() >= 0.05)
        return;
    if (Math.random() < 0.5 && field.blobs.length > 1) {
        // Remove a random uncounted blob (prefer uncounted to avoid breaking active counting)
        const uncounted = field.blobs.filter((b) => b.countedBy.size === 0 && !b.isCarried);
        if (uncounted.length > 0) {
            const victim = uncounted[Math.floor(Math.random() * uncounted.length)];
            const idx = field.blobs.indexOf(victim);
            field.blobs.splice(idx, 1);
            field.totalCount = field.blobs.length;
        }
    }
    else {
        // Add a blob at a random position
        const x = MARGIN + Math.random() * (WORLD_WIDTH - 2 * MARGIN);
        const y = MARGIN + Math.random() * (WORLD_HEIGHT - 2 * MARGIN);
        const BLOB_COLORS = [
            "#F9A8D4", "#FCA5A5", "#FDBA74", "#FCD34D",
            "#86EFAC", "#67E8F9", "#A5B4FC", "#C4B5FD",
        ];
        const newId = field.blobs.length > 0 ? Math.max(...field.blobs.map((b) => b.id)) + 1 : 0;
        field.blobs.push({
            id: newId,
            position: vec2(x, y),
            radius: 10 + Math.random() * 8,
            baseColor: BLOB_COLORS[newId % BLOB_COLORS.length],
            countedBy: new Set(),
            markedBy: null,
            isCarried: false,
            carriedBy: null,
            placedPosition: null,
            markGlow: 0,
            countFlash: 0,
            countFlashColor: "#ffffff",
        });
        field.totalCount = field.blobs.length;
    }
}
// --- Core env functions ---
export function resetEnv(config) {
    // Random blob count
    const blobCount = config.blobCountMin +
        Math.floor(Math.random() * (config.blobCountMax - config.blobCountMin + 1));
    // Random arrangement
    const arrangementType = randomArrangement();
    const positions = generateArrangement(blobCount, WORLD_WIDTH, WORLD_HEIGHT, MARGIN, MIN_SEPARATION, arrangementType);
    // Create blob field from positions
    const blobField = createBlobFieldFromPositions(positions);
    // Select bot personality based on stage
    const personality = selectBotPersonality(config.stage);
    // Random bot start position
    const startX = MARGIN + Math.random() * (WORLD_WIDTH - 2 * MARGIN);
    const startY = MARGIN + Math.random() * (WORLD_HEIGHT - 2 * MARGIN);
    const startPos = vec2(startX, startY);
    // Waypoints are blob positions
    const waypoints = blobField.blobs.map((b) => b.position);
    const bot = createBot(personality.name, personality, startPos, waypoints);
    // Wire up strategy or arrival handler
    let strategy = null;
    if (personality.name === "organizing" || personality.name === "grid") {
        strategy = createStrategy();
        bot.onArrival = null;
    }
    else {
        bot.onArrival = createArrivalHandler(personality.name, blobField);
    }
    return {
        bot,
        blobField,
        strategy,
        phase: "counting",
        step: 0,
        done: false,
        truncated: false,
        botTotalDistance: 0,
        prevBotPos: { ...startPos },
        blobCountAtStart: blobField.blobs.length,
        botType: personality.name,
        arrangementType,
        reward: 0,
    };
}
export function stepEnv(state, config, action) {
    // If in predict phase, compute reward from action
    if (state.phase === "predict") {
        const botTally = state.bot.countTally;
        let reward = 0;
        if (action !== null) {
            if (action === botTally) {
                reward = 1.0;
            }
            else if (Math.abs(action - botTally) <= 1) {
                reward = 0.5;
            }
        }
        state.done = true;
        state.reward = reward;
        return {
            obs: getObservation(state),
            reward,
            done: true,
            info: buildInfo(state, config),
        };
    }
    // --- Counting phase: advance simulation one tick ---
    state.step++;
    // Conservation OFF: maybe add/remove blobs
    if (!config.conservation) {
        mutateBlobs(state.blobField);
        // Update waypoints if blobs changed (bot still targets original positions,
        // which is intentional — confused bot tracks what IT counted)
    }
    // Update blob animations (mark glow, count flash — tracked even without rendering)
    updateBlobAnimations(state.blobField.blobs, DT);
    // Tick strategy for organizing/grid bots
    const bot = state.bot;
    if (!bot.allVisited) {
        const strategy = state.strategy;
        if (strategy) {
            if (strategy.phase === "done") {
                bot.allVisited = true;
                bot.dynamicTarget = null;
            }
            else {
                const placementFn = bot.personality.name === "grid"
                    ? computeGridPlacement
                    : computeLinePlacement;
                const strategyArrivalRadius = Math.max(20, bot.personality.waypointArrivalRadius);
                const target = tickStrategy(bot, state.blobField, strategy, WORLD_WIDTH, WORLD_HEIGHT, placementFn, strategyArrivalRadius);
                bot.dynamicTarget = target;
            }
        }
        // Update bot (no other bots — empty separation array)
        updateBot(bot, [], DT);
        // World boundary push
        if (bot.position.x < BOUNDARY_MARGIN)
            bot.velocity.x += 0.3;
        if (bot.position.x > WORLD_WIDTH - BOUNDARY_MARGIN)
            bot.velocity.x -= 0.3;
        if (bot.position.y < BOUNDARY_MARGIN)
            bot.velocity.y += 0.3;
        if (bot.position.y > WORLD_HEIGHT - BOUNDARY_MARGIN)
            bot.velocity.y -= 0.3;
    }
    // Update carried blobs
    updateCarriedBlobs(state.blobField.blobs, (botId) => {
        return bot.id === botId ? bot.position : null;
    });
    // Track distance
    const dx = bot.position.x - state.prevBotPos.x;
    const dy = bot.position.y - state.prevBotPos.y;
    state.botTotalDistance += Math.sqrt(dx * dx + dy * dy);
    state.prevBotPos = { ...bot.position };
    // Check if bot finished → transition to predict phase
    if (bot.allVisited) {
        state.phase = "predict";
    }
    // Check truncation (safety valve)
    if (state.step >= config.maxSteps) {
        state.truncated = true;
        state.done = true;
        return {
            obs: getObservation(state),
            reward: 0,
            done: true,
            info: buildInfo(state, config),
        };
    }
    return {
        obs: getObservation(state),
        reward: 0,
        done: false,
        info: buildInfo(state, config),
    };
}
function buildInfo(state, config) {
    return {
        bot_tally: state.bot.countTally,
        blob_count_start: state.blobCountAtStart,
        blob_count_end: state.blobField.blobs.length,
        bot_type: state.botType,
        episode_length: state.step,
        bot_distance: state.botTotalDistance,
        arrangement_type: state.arrangementType,
        phase: state.phase,
        conservation: config.conservation,
        stage: config.stage,
        truncated: state.truncated,
    };
}
