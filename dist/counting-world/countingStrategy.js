// --- Counting Strategy State Machines ---
// Each strategy bot (organizing, grid) has a state machine that drives
// its behavior through phases: gather blobs → arrange them → count them.
// The confused/marking/unconventional bots use simpler callback strategies.
import { pickUpBlob, placeBlob, countBlob } from "./blob.js";
import { vec2, dist } from "./steering.js";
export function createStrategy() {
    return {
        phase: "seeking",
        arrangementOrder: [],
        placementTargets: [],
        placedCount: 0,
        currentIndex: 0,
        countingIndex: 0,
        arrangedCount: 0,
    };
}
// --- Placement target generators ---
/**
 * Line formation: all blobs in a neat horizontal line.
 * Used by the organizing bot.
 */
export function computeLinePlacement(blobCount, canvasWidth, canvasHeight) {
    const targets = [];
    const spacing = 45;
    const totalWidth = (blobCount - 1) * spacing;
    const startX = (canvasWidth - totalWidth) / 2;
    const y = canvasHeight * 0.5; // middle of canvas
    for (let i = 0; i < blobCount; i++) {
        targets.push(vec2(startX + i * spacing, y));
    }
    return targets;
}
/**
 * Grid formation: blobs in rows of 5 (like tally marks / ten-frames).
 * Used by the grid bot.
 */
export function computeGridPlacement(blobCount, canvasWidth, canvasHeight) {
    const targets = [];
    const colSpacing = 45;
    const rowSpacing = 45;
    const cols = 5;
    const rows = Math.ceil(blobCount / cols);
    const totalWidth = (Math.min(blobCount, cols) - 1) * colSpacing;
    const totalHeight = (rows - 1) * rowSpacing;
    const startX = (canvasWidth - totalWidth) / 2;
    const startY = (canvasHeight - totalHeight) / 2;
    for (let i = 0; i < blobCount; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        targets.push(vec2(startX + col * colSpacing, startY + row * rowSpacing));
    }
    return targets;
}
/**
 * Determine which blob to pick up next.
 * For organizing bot: nearest unplaced blob.
 * For grid bot: nearest unplaced blob.
 */
function findNearestUnplacedBlob(botPosition, field, strategy) {
    let bestIdx = -1;
    let bestDist = Infinity;
    const placedSet = new Set(strategy.arrangementOrder.slice(0, strategy.placedCount));
    for (let i = 0; i < field.blobs.length; i++) {
        if (placedSet.has(i))
            continue;
        if (field.blobs[i].isCarried)
            continue;
        const d = dist(botPosition, field.blobs[i].position);
        if (d < bestDist) {
            bestDist = d;
            bestIdx = i;
        }
    }
    return bestIdx;
}
// --- Main strategy tick ---
/**
 * Tick the counting strategy state machine. Called each frame.
 * Returns the waypoint the bot should navigate to (or null if done).
 * Mutates strategy and blob states as the bot progresses.
 */
export function tickStrategy(bot, field, strategy, canvasWidth, canvasHeight, placementFn, arrivalRadius) {
    // Initialize placement targets on first call
    if (strategy.placementTargets.length === 0) {
        strategy.placementTargets = placementFn(field.blobs.length, canvasWidth, canvasHeight);
    }
    switch (strategy.phase) {
        case "seeking": {
            // Find nearest unplaced blob
            const blobIdx = findNearestUnplacedBlob(bot.position, field, strategy);
            if (blobIdx === -1) {
                // Check if we've truly placed all blobs, or if some are just carried by others
                if (strategy.placedCount >= field.blobs.length) {
                    // All blobs placed by this strategy — move to counting
                    strategy.phase = "counting";
                    strategy.countingIndex = 0;
                    return strategy.placementTargets[0] ?? bot.position;
                }
                // Some blobs still carried by other bots — wait
                return bot.position;
            }
            const blob = field.blobs[blobIdx];
            const d = dist(bot.position, blob.position);
            if (d < arrivalRadius) {
                // Close enough to pick up
                if (pickUpBlob(blob, bot.id)) {
                    strategy.arrangementOrder[strategy.placedCount] = blobIdx;
                    strategy.phase = "carrying";
                    blob.placedPosition = strategy.placementTargets[strategy.placedCount];
                    return strategy.placementTargets[strategy.placedCount];
                }
            }
            // Navigate to the blob
            return blob.position;
        }
        case "carrying": {
            // Navigate to placement target
            const target = strategy.placementTargets[strategy.placedCount];
            const d = dist(bot.position, target);
            if (d < arrivalRadius) {
                // Place the blob
                const blobIdx = strategy.arrangementOrder[strategy.placedCount];
                const blob = field.blobs[blobIdx];
                placeBlob(blob);
                strategy.placedCount++;
                strategy.phase = "seeking";
                // Stay put — next tick will resolve seeking target
                return bot.position;
            }
            return target;
        }
        case "counting": {
            // Walk along the arrangement counting each blob
            if (strategy.countingIndex >= strategy.placedCount) {
                strategy.phase = "done";
                return bot.position;
            }
            const blobIdx = strategy.arrangementOrder[strategy.countingIndex];
            const blob = field.blobs[blobIdx];
            const d = dist(bot.position, blob.position);
            if (d < arrivalRadius) {
                // Count this blob
                countBlob(blob, bot.id, bot.personality.color);
                strategy.arrangedCount++;
                bot.countTally = strategy.arrangedCount;
                strategy.countingIndex++;
                // Navigate to next blob in arrangement
                if (strategy.countingIndex < strategy.placedCount) {
                    return field.blobs[strategy.arrangementOrder[strategy.countingIndex]].position;
                }
                strategy.phase = "done";
                return bot.position;
            }
            return blob.position;
        }
        case "done":
            return bot.position;
        default:
            return bot.position;
    }
}
