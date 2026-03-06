// --- Headless Counting World Environment ---
// Runs the same simulation logic as the browser version, without rendering.
// Grid-based counting: blobs physically slide from field to a 5×5 grid.
// Uncounting reverses in FILO order. The grid IS the count.
// Produces observation vectors for the RL agent.
import { ALL_PERSONALITIES, } from "./counting-world/botPersonalities.js";
import { createBot, updateBot, } from "./counting-world/bot.js";
import { vec2, dist } from "./counting-world/steering.js";
import { markBlob, updateBlobAnimations, } from "./counting-world/blob.js";
import { createGrid, countBlobToGrid, uncountBlobFromGrid, getNextUncountTarget, GRID_COLS, GRID_ROWS, } from "./counting-world/gridCounting.js";
import { randomArrangement, generateArrangement, } from "./arrangements.js";
// --- Constants ---
const WORLD_WIDTH = 1400;
const WORLD_HEIGHT = 1000;
const MARGIN = 120;
const MIN_SEPARATION = 50;
const DT = 1;
const MAX_BLOB_COUNT = 25; // observation vector padding
const BOUNDARY_MARGIN = 40;
const UNCOUNT_ARRIVAL_RADIUS = 30;
const UNCOUNT_PAUSE_FRAMES = 15;
export const DEFAULT_CONFIG = {
    stage: 1,
    conservation: true,
    blobCountMin: 6,
    blobCountMax: 25,
    maxSteps: 5000,
    bidirectional: false,
};
// --- Observation vector ---
// [bot_x, bot_y, bot_state, blob_0_x, blob_0_y, ..., blob_24_x, blob_24_y,
//  blob_0_grid_slot, ..., blob_24_grid_slot, actual_blob_count, phase_indicator,
//  grid_filled_normalized, grid_filled_raw]
// Total: 2 + 1 + 50 + 25 + 1 + 1 + 1 + 1 = 82
export const OBS_SIZE = 82;
function botStateValue(bot) {
    if (bot.allVisited)
        return 2; // done
    if (bot.pauseTimer > 0)
        return 1; // counting/paused at blob
    return 0; // seeking
}
export function getObservation(state) {
    const obs = new Array(OBS_SIZE).fill(0);
    const { bot, blobField, grid, phase } = state;
    // Normalize positions to [0, 1]
    obs[0] = bot.position.x / WORLD_WIDTH;
    obs[1] = bot.position.y / WORLD_HEIGHT;
    obs[2] = botStateValue(bot);
    // Blob positions (padded to MAX_BLOB_COUNT) — NOW DYNAMIC as blobs animate to/from grid
    const blobOffset = 3;
    for (let i = 0; i < blobField.blobs.length && i < MAX_BLOB_COUNT; i++) {
        obs[blobOffset + i * 2] = blobField.blobs[i].position.x / WORLD_WIDTH;
        obs[blobOffset + i * 2 + 1] = blobField.blobs[i].position.y / WORLD_HEIGHT;
    }
    // Grid slot assignments (replaces mark flags): normalized slot index if gridded, 0 if in field
    const slotOffset = blobOffset + MAX_BLOB_COUNT * 2; // 3 + 50 = 53
    const maxSlots = GRID_COLS * GRID_ROWS;
    for (let i = 0; i < blobField.blobs.length && i < MAX_BLOB_COUNT; i++) {
        const blob = blobField.blobs[i];
        obs[slotOffset + i] = blob.gridSlot !== null
            ? (blob.gridSlot + 1) / maxSlots
            : 0;
    }
    // Actual blob count this episode
    obs[slotOffset + MAX_BLOB_COUNT] = blobField.blobs.length; // index 78
    // Phase indicator: 0 = counting, 0.5 = unmarking, 1 = predict
    obs[slotOffset + MAX_BLOB_COUNT + 1] =
        phase === "predict" ? 1 : phase === "unmarking" ? 0.5 : 0; // index 79
    // Grid filled count (normalized) — index 80
    obs[slotOffset + MAX_BLOB_COUNT + 2] = grid.filledCount / MAX_BLOB_COUNT;
    // Grid filled count (raw integer) — index 81
    obs[slotOffset + MAX_BLOB_COUNT + 3] = grid.filledCount;
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
// --- Grid arrival handlers ---
function createGridArrivalHandler(personalityName, field, grid) {
    switch (personalityName) {
        case "confused": {
            const stopAfter = field.totalCount + 3 + Math.floor(Math.random() * 4);
            return (bot, idx) => {
                const blob = field.blobs[idx];
                if (!blob)
                    return true;
                // Already gridded → skip (don't count)
                if (blob.gridSlot !== null)
                    return false;
                const placed = countBlobToGrid(blob, idx, grid, bot.id, bot.personality.color);
                if (placed) {
                    const forgets = Math.random() < 0.15;
                    if (bot.countTally >= stopAfter) {
                        for (let i = 0; i < bot.waypoints.length; i++) {
                            bot.visitedIndices.add(i);
                        }
                        return true;
                    }
                    return !forgets;
                }
                return false;
            };
        }
        case "marking":
            return (bot, idx) => {
                const blob = field.blobs[idx];
                if (!blob)
                    return true;
                const placed = countBlobToGrid(blob, idx, grid, bot.id, bot.personality.color);
                if (placed) {
                    markBlob(blob, bot.id);
                }
                return placed;
            };
        case "organizing":
        case "grid":
            return (bot, idx) => {
                const blob = field.blobs[idx];
                if (!blob)
                    return true;
                const placed = countBlobToGrid(blob, idx, grid, bot.id, bot.personality.color);
                return placed;
            };
        case "unconventional":
            return (bot, idx) => {
                const blob = field.blobs[idx];
                if (!blob)
                    return true;
                const placed = countBlobToGrid(blob, idx, grid, bot.id, bot.personality.color);
                return placed;
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
        fieldPosition: { ...pos },
        gridSlot: null,
        animatingTo: null,
        animatingFrom: null,
        animProgress: 0,
        markGlow: 0,
        countFlash: 0,
        countFlashColor: "#ffffff",
    }));
    return { blobs, totalCount: blobs.length };
}
// --- Conservation OFF: blob mutation ---
function mutateBlobs(field, grid) {
    // 5% chance per step of adding or removing one blob
    if (Math.random() >= 0.05)
        return;
    if (Math.random() < 0.5 && field.blobs.length > 1) {
        // Remove a random uncounted blob that is NOT in the grid
        const uncounted = field.blobs.filter((b) => b.countedBy.size === 0 && !b.isCarried && b.gridSlot === null);
        if (uncounted.length > 0) {
            const victim = uncounted[Math.floor(Math.random() * uncounted.length)];
            const idx = field.blobs.indexOf(victim);
            field.blobs.splice(idx, 1);
            field.totalCount = field.blobs.length;
        }
    }
    else {
        // Add a blob at a random position in the field zone
        const fieldMaxX = WORLD_WIDTH * 0.55;
        const x = MARGIN + Math.random() * (fieldMaxX - 2 * MARGIN);
        const y = MARGIN + Math.random() * (WORLD_HEIGHT - 2 * MARGIN);
        const BLOB_COLORS = [
            "#F9A8D4", "#FCA5A5", "#FDBA74", "#FCD34D",
            "#86EFAC", "#67E8F9", "#A5B4FC", "#C4B5FD",
        ];
        const newId = field.blobs.length > 0 ? Math.max(...field.blobs.map((b) => b.id)) + 1 : 0;
        const pos = vec2(x, y);
        field.blobs.push({
            id: newId,
            position: pos,
            radius: 10 + Math.random() * 8,
            baseColor: BLOB_COLORS[newId % BLOB_COLORS.length],
            countedBy: new Set(),
            markedBy: null,
            isCarried: false,
            carriedBy: null,
            placedPosition: null,
            fieldPosition: { ...pos },
            gridSlot: null,
            animatingTo: null,
            animatingFrom: null,
            animProgress: 0,
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
    // Random arrangement — constrain to field zone (left 55% of world)
    const arrangementType = randomArrangement();
    const fieldWidth = Math.floor(WORLD_WIDTH * 0.55);
    const positions = generateArrangement(blobCount, fieldWidth, WORLD_HEIGHT, MARGIN, MIN_SEPARATION, arrangementType);
    // Create blob field from positions
    const blobField = createBlobFieldFromPositions(positions);
    // Create grid
    const grid = createGrid(WORLD_WIDTH, WORLD_HEIGHT);
    // Select bot personality based on stage
    const personality = selectBotPersonality(config.stage);
    // Random bot start position in field area
    const startX = MARGIN + Math.random() * (fieldWidth - 2 * MARGIN);
    const startY = MARGIN + Math.random() * (WORLD_HEIGHT - 2 * MARGIN);
    const startPos = vec2(startX, startY);
    // Waypoints are blob positions
    const waypoints = blobField.blobs.map((b) => b.position);
    const bot = createBot(personality.name, personality, startPos, waypoints);
    // All bots use grid arrival handler (no strategy FSM needed)
    bot.onArrival = createGridArrivalHandler(personality.name, blobField, grid);
    return {
        bot,
        blobField,
        grid,
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
        bidirectional: config.bidirectional,
        uncountPauseTimer: 0,
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
    // --- Advance simulation one tick ---
    state.step++;
    // Conservation OFF: maybe add/remove blobs (not during unmark phase — count must be stable)
    if (!config.conservation && state.phase !== "unmarking") {
        mutateBlobs(state.blobField, state.grid);
    }
    // Update blob animations (position transitions, mark glow, count flash)
    updateBlobAnimations(state.blobField.blobs, DT);
    const bot = state.bot;
    // --- Unmarking phase: bot navigates to grid blobs via dynamicTarget ---
    if (state.phase === "unmarking") {
        // Handle pause between uncounts
        if (state.uncountPauseTimer > 0) {
            state.uncountPauseTimer--;
            bot.velocity = { x: bot.velocity.x * 0.85, y: bot.velocity.y * 0.85 };
        }
        else {
            // Find next uncount target
            const target = getNextUncountTarget(state.blobField, state.grid);
            if (target) {
                bot.dynamicTarget = target;
                // Check arrival at target
                const d = dist(bot.position, target);
                if (d < UNCOUNT_ARRIVAL_RADIUS) {
                    uncountBlobFromGrid(state.blobField, state.grid, bot.id, WORLD_WIDTH, WORLD_HEIGHT);
                    state.uncountPauseTimer = UNCOUNT_PAUSE_FRAMES;
                    bot.dynamicTarget = null;
                }
            }
            // Check if grid is empty → transition to predict
            if (state.grid.filledCount === 0) {
                state.phase = "predict";
            }
        }
        // Update bot movement
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
    // --- Counting phase ---
    else if (!bot.allVisited) {
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
    // Track distance
    const dx = bot.position.x - state.prevBotPos.x;
    const dy = bot.position.y - state.prevBotPos.y;
    state.botTotalDistance += Math.sqrt(dx * dx + dy * dy);
    state.prevBotPos = { ...bot.position };
    // Check if bot finished counting → transition to unmark or predict phase
    if (bot.allVisited && state.phase === "counting") {
        if (state.bidirectional) {
            // Enter unmark phase: bot navigates to grid blobs via dynamicTarget
            state.phase = "unmarking";
            bot.allVisited = false;
            bot.dynamicTarget = null;
            bot.visitedIndices.clear();
            // No waypoints needed — we drive via dynamicTarget + getNextUncountTarget
            bot.waypoints = [];
        }
        else {
            state.phase = "predict";
        }
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
        bidirectional: config.bidirectional,
        grid_filled: state.grid.filledCount,
    };
}
