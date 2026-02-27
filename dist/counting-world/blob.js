// --- Blob Entity & Field Generator ---
// Blobs are the countable objects in the world. Conservation law: they never
// appear or vanish. The total count is fixed for a given level.
import { vec2, dist } from "./steering.js";
// --- Blob colors ---
// Soft, organic colors that feel like counting objects
const BLOB_COLORS = [
    "#F9A8D4", // pink
    "#FCA5A5", // light red
    "#FDBA74", // orange
    "#FCD34D", // yellow
    "#86EFAC", // light green
    "#67E8F9", // cyan
    "#A5B4FC", // indigo
    "#C4B5FD", // violet
];
const DEFAULT_CONFIG = {
    count: 10,
    width: 800,
    height: 600,
    margin: 80,
    minSeparation: 35,
    minRadius: 10,
    maxRadius: 18,
    clusterProbability: 0.3,
    clusterRadius: 60,
};
export function generateBlobField(config = {}) {
    const c = { ...DEFAULT_CONFIG, ...config };
    const blobs = [];
    let attempts = 0;
    const maxAttempts = c.count * 50;
    while (blobs.length < c.count && attempts < maxAttempts) {
        attempts++;
        let x, y;
        // Sometimes cluster near an existing blob
        if (blobs.length > 0 && Math.random() < c.clusterProbability) {
            const anchor = blobs[Math.floor(Math.random() * blobs.length)];
            const angle = Math.random() * Math.PI * 2;
            const r = c.minSeparation + Math.random() * (c.clusterRadius - c.minSeparation);
            x = anchor.position.x + Math.cos(angle) * r;
            y = anchor.position.y + Math.sin(angle) * r;
        }
        else {
            x = c.margin + Math.random() * (c.width - 2 * c.margin);
            y = c.margin + Math.random() * (c.height - 2 * c.margin);
        }
        // Bounds check
        if (x < c.margin || x > c.width - c.margin || y < c.margin || y > c.height - c.margin) {
            continue;
        }
        const candidate = vec2(x, y);
        const tooClose = blobs.some((b) => dist(b.position, candidate) < c.minSeparation);
        if (tooClose)
            continue;
        const radius = c.minRadius + Math.random() * (c.maxRadius - c.minRadius);
        const colorIdx = blobs.length % BLOB_COLORS.length;
        blobs.push({
            id: blobs.length,
            position: candidate,
            radius,
            baseColor: BLOB_COLORS[colorIdx],
            countedBy: new Set(),
            markedBy: null,
            isCarried: false,
            carriedBy: null,
            placedPosition: null,
            markGlow: 0,
            countFlash: 0,
            countFlashColor: "#ffffff",
        });
    }
    return {
        blobs,
        totalCount: blobs.length,
    };
}
// --- Blob interaction ---
/**
 * A bot counts a blob. Returns true if this is the first time this bot counted it.
 */
export function countBlob(blob, botId, botColor) {
    const isNew = !blob.countedBy.has(botId);
    blob.countedBy.add(botId);
    blob.countFlash = 1.0;
    blob.countFlashColor = botColor;
    return isNew;
}
/**
 * A bot marks a blob (marking bot only). Persistent visual glow.
 */
export function markBlob(blob, botId) {
    blob.markedBy = botId;
    // Glow animates toward 1.0 in the render loop
}
/**
 * A bot unmarks a blob (bidirectional episode: unmark phase).
 * Clears both the persistent mark and the countedBy tracking,
 * producing a -1 delta in the observation vector mark slots.
 */
export function unmarkBlob(blob, botId) {
    if (!blob.countedBy.has(botId))
        return false;
    blob.countedBy.delete(botId);
    if (blob.markedBy === botId) {
        blob.markedBy = null;
        blob.markGlow = 0;
    }
    blob.countFlash = 1.0; // visual feedback
    return true;
}
/**
 * Update blob animation state each frame.
 */
export function updateBlobAnimations(blobs, dt) {
    for (const blob of blobs) {
        // Count flash decays
        if (blob.countFlash > 0) {
            blob.countFlash = Math.max(0, blob.countFlash - 0.04 * dt);
        }
        // Mark glow fades in
        if (blob.markedBy && blob.markGlow < 1) {
            blob.markGlow = Math.min(1, blob.markGlow + 0.03 * dt);
        }
    }
}
/**
 * Pick up a blob (organizing/grid bot). Returns true if successful.
 */
export function pickUpBlob(blob, botId) {
    if (blob.isCarried)
        return false; // already carried
    blob.isCarried = true;
    blob.carriedBy = botId;
    return true;
}
/**
 * Place a carried blob at its target position. Returns true if successful.
 */
export function placeBlob(blob) {
    if (!blob.isCarried)
        return false;
    if (blob.placedPosition) {
        blob.position = { ...blob.placedPosition };
    }
    blob.isCarried = false;
    blob.carriedBy = null;
    blob.placedPosition = null;
    return true;
}
/**
 * Update carried blob positions to follow their carrier.
 */
export function updateCarriedBlobs(blobs, getBotPosition) {
    for (const blob of blobs) {
        if (blob.isCarried && blob.carriedBy) {
            const botPos = getBotPosition(blob.carriedBy);
            if (botPos) {
                // Float slightly above/behind the bot
                blob.position = { x: botPos.x, y: botPos.y - 15 };
            }
        }
    }
}
/**
 * Reset all blob counting state (for demo restart).
 */
export function resetBlobField(field) {
    for (const blob of field.blobs) {
        blob.countedBy.clear();
        blob.markedBy = null;
        blob.isCarried = false;
        blob.carriedBy = null;
        blob.placedPosition = null;
        blob.markGlow = 0;
        blob.countFlash = 0;
    }
}
/**
 * Get how many blobs a specific bot has counted.
 */
export function getBotCount(field, botId) {
    return field.blobs.filter((b) => b.countedBy.has(botId)).length;
}
/**
 * Get how many unique blobs a bot has counted (excludes recounts).
 * For confused bot, this differs from the bot's internal count.
 */
export function getBotUniqueCount(field, botId) {
    return field.blobs.filter((b) => b.countedBy.has(botId)).length;
}
